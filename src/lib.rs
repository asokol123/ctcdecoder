mod tree;
mod vec2d;

use pyo3::exceptions::PyRuntimeError;
use numpy::array::PyArray2;

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::types::PyString;
use tree::*;

#[derive(Clone, Copy, Debug)]
struct SearchPoint {
    /// The node search should progress from.
    node: i32,
    /// The transition state for crf.
    state: usize,
    /// The cumulative probability of the labelling so far for paths without any leading blank
    /// labels.
    label_prob: f32,
    /// The cumulative probability of the labelling so far for paths with one or more leading
    /// blank labels.
    gap_prob: f32,
}

impl SearchPoint {
    /// The total probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels.
    fn probability(&self) -> f32 {
        self.label_prob + self.gap_prob
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SearchError {
    RanOutOfBeam,
    IncomparableValues,
    InvalidEnvelope,
}


impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::RanOutOfBeam => {
                write!(f, "Ran out of search space (beam_cut_threshold too high)")
            }
            SearchError::IncomparableValues => {
                write!(f, "Failed to compare values (NaNs in input?)")
            }
            // TODO: document envelope constraints
            SearchError::InvalidEnvelope => write!(f, "Invalid envelope values"),
        }
    }
}

#[pymodule]
fn ctcdecoder(_py: Python<'_>, _m: &PyModule) -> PyResult<()> {
    #[pyfn(_m)]
    #[pyo3(name = "beam_search")]
    fn beam_search<'py>(
        _py: Python<'py>,
        probs: &PyArray2<f32>,
        alphabet: &PyString,
        beam_size: usize,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        assert_eq!(
            probs.shape().len(),
            2,
            "Expected 2d tensor, got {}",
            probs.shape().len()
        );

        let alphabet = alphabet.to_str()?;

            let probs = unsafe { probs.as_array() };

            let bs: PyResult<(Vec<String>, Vec<f32>)> = {
                let network_output = probs;
                let beam_cut_threshold = 0.;

                // alphabet size minus the blank label
                let alphabet_size = alphabet.len() - 1;

                let mut suffix_tree = SuffixTree::new(alphabet_size);
                let mut beam = vec![SearchPoint {
                    node: ROOT_NODE,
                    state: 0,
                    gap_prob: 1.0,
                    label_prob: 0.0,
                }];
                let mut next_beam = Vec::new();

                for (idx, pr) in network_output.outer_iter().enumerate() {
                    next_beam.clear();

                    for &SearchPoint {
                        node,
                        label_prob,
                        gap_prob,
                        state,
                    } in &beam
                    {
                        let tip_label = suffix_tree.label(node);

                        // add N to beam
                        if pr[0] > beam_cut_threshold {
                            next_beam.push(SearchPoint {
                                node,
                                state,
                                label_prob: 0.0,
                                gap_prob: (label_prob + gap_prob) * pr[0],
                            });
                        }

                        for (label, pr_b) in pr.iter().skip(1).enumerate() {
                            if pr_b < &beam_cut_threshold {
                                continue;
                            }

                            if Some(label) == tip_label {
                                next_beam.push(SearchPoint {
                                    node,
                                    label_prob: label_prob * pr_b,
                                    gap_prob: 0.0,
                                    state,
                                });
                                let new_node_idx =
                                    suffix_tree.get_child(node, label).or_else(|| {
                                        if gap_prob > 0.0 {
                                            Some(suffix_tree.add_node(node, label, idx))
                                        } else {
                                            None
                                        }
                                    });

                                if let Some(idx) = new_node_idx {
                                    next_beam.push(SearchPoint {
                                        node: idx,
                                        state,
                                        label_prob: gap_prob * pr_b,
                                        gap_prob: 0.0,
                                    });
                                }
                            } else {
                                let new_node_idx = suffix_tree
                                    .get_child(node, label)
                                    .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                                next_beam.push(SearchPoint {
                                    node: new_node_idx,
                                    state,
                                    label_prob: (label_prob + gap_prob) * pr_b,
                                    gap_prob: 0.0,
                                });
                            }
                        }
                    }
                    std::mem::swap(&mut beam, &mut next_beam);

                    const DELETE_MARKER: i32 = i32::MIN;
                    beam.sort_by_key(|x| x.node);
                    let mut last_key = DELETE_MARKER;
                    let mut last_key_pos = 0;
                    for i in 0..beam.len() {
                        let beam_item = beam[i];
                        if beam_item.node == last_key {
                            beam[last_key_pos].label_prob += beam_item.label_prob;
                            beam[last_key_pos].gap_prob += beam_item.gap_prob;
                            beam[i].node = DELETE_MARKER;
                        } else {
                            last_key_pos = i;
                            last_key = beam_item.node;
                        }
                    }

                    beam.retain(|x| x.node != DELETE_MARKER);
                    let mut has_nans = false;
                    beam.sort_unstable_by(|a, b| {
                        (b.probability())
                            .partial_cmp(&(a.probability()))
                            .unwrap_or_else(|| {
                                has_nans = true;
                                std::cmp::Ordering::Equal // don't really care
                            })
                    });
                    if has_nans {
                        return Err(PyRuntimeError::new_err(format!("{}", SearchError::IncomparableValues)));
                    }
                    beam.truncate(beam_size);
                    if beam.is_empty() {
                        // we've run out of beam (probably the threshold is too high)
                        return Err(PyRuntimeError::new_err(format!("{}", SearchError::RanOutOfBeam)));
                    }
                    let top = beam[0].probability();
                    for mut x in &mut beam {
                        x.label_prob /= top;
                        x.gap_prob /= top;
                    }
                }

                let mut probas = Vec::new();
                let mut sequences = Vec::new();

                beam.drain(..).for_each(|beam| {
                    if beam.node != ROOT_NODE {
                        probas.push(beam.probability());

                        let mut sequence = String::new();
                        for (label, &time) in suffix_tree.iter_from(beam.node) {
                            sequence.push(alphabet.as_bytes()[label + 1] as char);
                        }

                        sequences.push(sequence.chars().rev().collect::<String>());
                    }
                });

                Ok((sequences, probas))
            };

            bs
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
