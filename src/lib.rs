mod tree;
mod vec2d;

use numpy::array::PyArray2;
use pyo3::exceptions::{PyAssertionError, PyRuntimeError};

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::types::{PyFloat, PyString};
use pyo3::{PyAny, PyObject};
use tree::*;

#[derive(Clone, Copy, Debug)]
struct SearchPoint {
    /// The node search should progress from.
    node: i32,
    prob: f32,
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

fn get_lm_prob(
    path: &str,
    i: usize,
    lm_model: Option<&PyAny>,
    lm_alpha: f32,
    lm_beta: f32,
) -> PyResult<f32> {
    if let Some(lm) = lm_model {
        Ok(lm_alpha
            * (lm
                .call_method1("score", (path,))?
                .downcast::<PyFloat>()?
                .value() as f32)
                .exp()
            + lm_beta * (i as f32))
    } else {
        Ok(0_f32)
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
        lm_model: Option<&PyAny>,
        lm_alpha: f32,
        lm_beta: f32,
    ) -> PyResult<Vec<(String, f32)>> {
        assert_eq!(
            probs.shape().len(),
            2,
            "Expected 2d array, got {}",
            probs.shape().len()
        );

        let alphabet = alphabet.to_str()?;

        let probs = unsafe { probs.as_array() };

        // alphabet size minus the blank label
        let alphabet_size = alphabet.len();
        if probs.shape()[1] != alphabet_size {
            return Err(PyAssertionError::new_err(format!(
                "Expected props.shape[1] ({}) == alphabet size ({})",
                probs.shape()[1],
                alphabet_size
            )));
        }

        let mut suffix_tree = SuffixTree::new(alphabet_size);
        let mut beam = vec![SearchPoint {
            node: ROOT_NODE,
            prob: 1.0,
        }];
        let mut next_beam = Vec::new();

        for (idx, pr) in probs.outer_iter().enumerate() {
            next_beam.clear();

            for &SearchPoint { node, prob } in beam.iter() {
                let tip_label = suffix_tree.label(node);

                let mut curr_path = suffix_tree.get_path(node, alphabet);

                next_beam.push(SearchPoint {
                    node,
                    prob: prob * pr[0] + get_lm_prob(&curr_path, idx, lm_model, lm_alpha, lm_beta)?,
                });

                for (label, pr_b) in pr.iter().skip(1).enumerate() {
                    if Some(label) == tip_label {
                        next_beam.push(SearchPoint {
                            node,
                            prob: prob * pr_b
                                + get_lm_prob(&curr_path, idx, lm_model, lm_alpha, lm_beta)?,
                        });
                    } else {
                        curr_path.push(alphabet.as_bytes()[label] as char);
                        let new_node_idx = suffix_tree
                            .get_child(node, label)
                            .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                        next_beam.push(SearchPoint {
                            node: new_node_idx,
                            prob: prob * pr_b
                                + get_lm_prob(&curr_path, idx, lm_model, lm_alpha, lm_beta)?,
                        });

                        curr_path.pop();
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
                    beam[last_key_pos].prob += beam_item.prob;
                    beam[i].node = DELETE_MARKER;
                } else {
                    last_key_pos = i;
                    last_key = beam_item.node;
                }
            }

            beam.retain(|x| x.node != DELETE_MARKER);
            let mut has_nans = false;
            beam.sort_unstable_by(|a, b| {
                (b.prob).partial_cmp(&(a.prob)).unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal // don't really care
                })
            });
            if has_nans {
                return Err(PyRuntimeError::new_err(format!(
                    "{}",
                    SearchError::IncomparableValues
                )));
            }
            beam.truncate(beam_size);
            if beam.is_empty() {
                // we've run out of beam (probably the threshold is too high)
                return Err(PyRuntimeError::new_err(format!(
                    "{}",
                    SearchError::RanOutOfBeam
                )));
            }
            let top = beam[0].prob;
            for mut x in &mut beam {
                x.prob /= top;
            }
        }

        let mut ans = Vec::new();

        beam.drain(..).for_each(|beam| {
            if beam.node != ROOT_NODE {
                ans.push((suffix_tree.get_path(beam.node, alphabet), beam.prob));
            }
        });

        Ok(ans)
    }

    Ok(())
}
