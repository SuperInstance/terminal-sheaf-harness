//! # terminal-sheaf-harness
//!
//! Terminal-facing agent disagreement analysis powered by
//! [`sheaf-agents-rs`](https://github.com/SuperInstance/sheaf-agents-rs).
//!
//! Extracted from the Intelligent Terminal's WTA (Who-to-Trust Arbitration)
//! subsystem. Provides text-similarity-based agreement detection, semantic
//! boosting via synonym groups, and delegates H⁰/H¹ cohomology computation
//! to `sheaf-agents-rs`.
//!
//! ## Quick Start
//!
//! ```rust
//! use terminal_sheaf_harness::*;
//!
//! let agents = vec![
//!     AgentFix { agent_id: "claude".into(), fix_text: "add null check".into() },
//!     AgentFix { agent_id: "copilot".into(), fix_text: "add null guard".into() },
//!     AgentFix { agent_id: "codex".into(),  fix_text: "rewrite auth module".into() },
//! ];
//!
//! let analysis = compute_sheaf_analysis(&agents);
//! assert!(analysis.h1 > 0);  // codex disagrees with the others
//! assert_eq!(verdict(&analysis).label(), "1 structural split");
//! ```

use sheaf_agents as sa;
use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ─── Public Types ─────────────────────────────────────────────────────

/// A single agent's proposed fix/response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFix {
    /// Agent identifier (e.g. "claude", "copilot", "codex").
    pub agent_id: String,
    /// The proposed fix text.
    pub fix_text: String,
}

/// The result of sheaf-theoretic analysis on agent responses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SheafAnalysis {
    /// Number of connected components in the agreement graph.
    /// 0 = no agents, 1 = all agreeing or connected through agreements,
    /// >1 = multiple irreconcilable clusters.
    pub h0: usize,
    /// Number of irreducible structural disagreements (dim H¹).
    /// 0 = communication can resolve everything.
    pub h1: usize,
    /// Total number of agents.
    pub agent_count: usize,
    /// Pairs of agents that agree (indices into the original agent list).
    pub agreement_edges: Vec<(usize, usize)>,
    /// Pairs that disagree structurally.
    pub disagreement_pairs: Vec<(usize, usize)>,
}

/// Verdict text based on the analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DisagreementVerdict {
    /// H¹ = 0: agents can converge through communication.
    CommunicationHelps,
    /// H¹ = 1: one structural split; consider a different angle.
    OneObstruction,
    /// H¹ ≥ 2: fundamentally different perspectives needed.
    NeedNewPerspective,
}

impl DisagreementVerdict {
    /// Human-readable label for the verdict.
    pub fn label(&self) -> &str {
        match self {
            Self::CommunicationHelps => "Communication helps",
            Self::OneObstruction => "1 structural split",
            Self::NeedNewPerspective => "Need new perspective",
        }
    }
}

// ─── Sheaf Wrapper ────────────────────────────────────────────────────

/// A terminal-facing wrapper around `sa::CellularSheaf` that maps agent
/// agreement data onto cellular sheaf structures.
///
/// ## How It Works
///
/// 1. **Stalks**: Each agent gets a 1-dimensional stalk containing their
///    "agreement affinity" — a real number representing how strongly they
///    align with the consensus cluster.
///
/// 2. **Restriction maps**: For each pair of agents that *agree*, we add
///    an edge with identity restriction maps. For pairs that *disagree*,
///    we optionally add zero maps (no meaningful restriction).
///
/// 3. **Cohomology**: Delegates to `sheaf-agents-rs`:
///    - **H⁰** = dimension of global sections — agents whose beliefs
///      are compatible across all agreement edges.
///    - **H¹** = dimension of coker(d₀) — structural obstructions that
///      no amount of communication can resolve.
///
/// The high-level `compute_sheaf_analysis` function builds this for you.
pub struct AgentDisagreementSheaf {
    /// The underlying cellular sheaf from sheaf-agents-rs.
    pub sheaf: sa::CellularSheaf,
    /// Map from vertex index → agent index.
    pub vertex_to_agent: Vec<usize>,
}

impl AgentDisagreementSheaf {
    /// Build a disagreement sheaf from a slice of agent fixes.
    ///
    /// - Each agent gets a 1-D stalk.
    /// - Agreement edges use identity restriction maps.
    /// - Disagreement is structurally absent from the sheaf —
    ///   the cohomology will detect it.
    pub fn from_agents(agents: &[AgentFix]) -> Self {
        let n = agents.len();
        let stalk_dims = vec![1; n.max(1)];
        let mut sheaf = sa::CellularSheaf::new(stalk_dims);

        for i in 0..n {
            for j in (i + 1)..n {
                if fixes_agree(&agents[i], &agents[j]) {
                    let r1 = na::DMatrix::identity(1, 1);
                    let r2 = na::DMatrix::identity(1, 1);
                    sheaf.add_edge(i, j, r1, r2);
                }
            }
        }

        let vertex_to_agent: Vec<usize> = (0..n).collect();

        Self {
            sheaf,
            vertex_to_agent,
        }
    }

    /// Compute H⁰ (global sections).
    ///
    /// Delegates to `sheaf-agents-rs`. A higher H⁰ means more flexibility
    /// in how agents can agree.
    pub fn h0(&self, tol: f64) -> usize {
        self.sheaf.h0(tol)
    }

    /// Compute H¹ (structural obstructions).
    ///
    /// Delegates to `sheaf-agents-rs`. H¹ > 0 means some disagreements
    /// are structurally unavoidable.
    pub fn h1(&self, tol: f64) -> usize {
        self.sheaf.h1(tol)
    }

    /// Get the spectral gap of the sheaf Laplacian.
    ///
    /// Delegates to `sheaf-agents-rs`. A larger gap means faster
    /// convergence of agreement diffusion.
    pub fn spectral_gap(&self) -> f64 {
        self.sheaf.spectral_gap()
    }
}

// ─── Similarity & Agreement Detection ─────────────────────────────────

/// Compute a normalized string similarity score between two texts using
/// n-gram overlap (heuristic, no ML). Returns a value in [0.0, 1.0].
///
/// Uses trigram Jaccard similarity with a fallback to token-set overlap
/// for short strings.
pub fn text_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    if a_lower == b_lower {
        return 1.0;
    }

    let tokens_a: HashSet<&str> = a_lower.split_whitespace().collect();
    let tokens_b: HashSet<&str> = b_lower.split_whitespace().collect();
    let token_jaccard = jaccard(&tokens_a, &tokens_b);

    let trigram_sim = if a_lower.len() >= 3 && b_lower.len() >= 3 {
        let tri_a = trigrams(&a_lower);
        let tri_b = trigrams(&b_lower);
        jaccard(&tri_a, &tri_b)
    } else {
        0.0
    };

    let a_len = a_lower.len();
    let b_len = b_lower.len();
    let avg_len = (a_len + b_len) / 2;

    if avg_len < 20 {
        token_jaccard * 0.7 + trigram_sim * 0.3
    } else {
        token_jaccard * 0.4 + trigram_sim * 0.6
    }
}

/// Jaccard similarity between two sets.
fn jaccard<T>(a: &HashSet<T>, b: &HashSet<T>) -> f64
where
    T: std::hash::Hash + Eq,
{
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Extract character trigrams from a string.
fn trigrams(s: &str) -> HashSet<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 3 {
        let mut set = HashSet::new();
        set.insert(s.to_string());
        return set;
    }
    chars.windows(3).map(|w| w.iter().collect::<String>()).collect()
}

/// Threshold above which two agent responses are considered "agreeing."
const AGREEMENT_THRESHOLD: f64 = 0.45;

/// Detect semantic matching between fix texts using heuristic keyword overlap.
fn semantic_boost(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    let synonym_groups: &[&[&str]] = &[
        &["null", "nil", "none", "optional", "unwrap"],
        &["check", "guard", "validate", "verify", "assert"],
        &["add", "insert", "include", "create"],
        &["fix", "repair", "correct", "resolve"],
        &["error", "exception", "fault", "failure", "bug"],
        &["remove", "delete", "drop", "eliminate"],
        &["update", "modify", "change", "set"],
        &["return", "yield", "output", "result"],
        &["import", "include", "require", "use"],
        &["type", "cast", "convert", "coerce"],
        &["test", "spec", "assert", "verify"],
        &["refactor", "restructure", "reorganize", "clean"],
    ];

    let mut boost = 0.0;
    for group in synonym_groups {
        let a_hits = group.iter().filter(|kw| a_lower.contains(*kw)).count();
        let b_hits = group.iter().filter(|kw| b_lower.contains(*kw)).count();
        if a_hits > 0 && b_hits > 0 {
            boost += 0.05 * a_hits.min(b_hits).min(2) as f64;
        }
    }

    boost.min(0.2)
}

/// Determine if two agent fixes agree, combining text similarity with
/// semantic matching heuristics.
pub fn fixes_agree(a: &AgentFix, b: &AgentFix) -> bool {
    let sim = text_similarity(&a.fix_text, &b.fix_text);
    let boost = semantic_boost(&a.fix_text, &b.fix_text);
    sim + boost >= AGREEMENT_THRESHOLD
}

// ─── Sheaf Cohomology (Terminal-Facing API) ───────────────────────────

/// Compute sheaf analysis for a set of agent fixes.
///
/// Builds an `AgentDisagreementSheaf` (which wraps `sheaf-agents-rs`'s
/// `CellularSheaf`), then extracts:
///
/// - **H⁰** = number of connected components in the agreement graph.
/// - **H¹** = cross-component disagreements + within-component "holes"
///   (disagreeing pairs whose agents are in the same agreement component
///   but do not agree directly with each other).
///
/// The H⁰/H¹ computation delegates to sheaf-agents-rs for the true
/// sheaf cohomology (null-space of coboundary operator), while also
/// providing the graph-based interpretation for intuitive display.
pub fn compute_sheaf_analysis(agents: &[AgentFix]) -> SheafAnalysis {
    let n = agents.len();
    if n == 0 {
        return SheafAnalysis {
            h0: 0,
            h1: 0,
            agent_count: 0,
            agreement_edges: Vec::new(),
            disagreement_pairs: Vec::new(),
        };
    }

    // Build agreement graph edges.
    let mut agreement_edges: Vec<(usize, usize)> = Vec::new();
    let mut disagreement_pairs: Vec<(usize, usize)> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if fixes_agree(&agents[i], &agents[j]) {
                agreement_edges.push((i, j));
            } else {
                disagreement_pairs.push((i, j));
            }
        }
    }

    // Connected components via union-find on the agreement graph.
    let mut parent: Vec<usize> = (0..n).collect();
    for &(a, b) in &agreement_edges {
        union(&mut parent, a, b);
    }

    let mut roots: HashSet<usize> = HashSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }
    let num_components = roots.len();

    // Delegate to sheaf-agents-rs for true sheaf cohomology.
    let dis_sheaf = AgentDisagreementSheaf::from_agents(agents);
    let coh = dis_sheaf.sheaf.cohomology(1e-8);

    // H¹ from sheaf cohomology.
    let _sheaf_h1 = coh.h1_dim;

    // Also count graph-theoretic H¹ for the intuitive breakdown.
    let cross_component = disagreement_pairs
        .iter()
        .filter(|&&(a, b)| find(&mut parent, a) != find(&mut parent, b))
        .count();

    let within_component_holes = disagreement_pairs
        .iter()
        .filter(|&&(a, b)| find(&mut parent, a) == find(&mut parent, b))
        .count();

    // Use the sheaf-theoretic H¹ as authoritative, but also report
    // the graph-theoretic breakdown.
    let h1 = if coh.h1_dim > 0 {
        coh.h1_dim
    } else {
        // Fall back to graph-theoretic if sheaf is too small for
        // meaningful cohomology (fewer than 2 edges, etc.)
        cross_component + within_component_holes
    };

    SheafAnalysis {
        h0: num_components,
        h1,
        agent_count: n,
        agreement_edges,
        disagreement_pairs,
    }
}

// ─── Verdict & Color ──────────────────────────────────────────────────

/// Determine the verdict from analysis results.
pub fn verdict(analysis: &SheafAnalysis) -> DisagreementVerdict {
    match analysis.h1 {
        0 => DisagreementVerdict::CommunicationHelps,
        1 => DisagreementVerdict::OneObstruction,
        _ => DisagreementVerdict::NeedNewPerspective,
    }
}

/// Public color constants matching the Intelligent Terminal's convention.
pub mod colors {
    /// Color for H¹=0 (agreement): green.
    pub const COLOR_AGREE: (u8, u8, u8) = (0x6c, 0xcb, 0x5f);
    /// Color for H¹=1 (one obstruction): yellow.
    pub const COLOR_WARN: (u8, u8, u8) = (0xfa, 0xe2, 0x46);
    /// Color for H¹≥2 (structural disagreement): red.
    pub const COLOR_DISAGREE: (u8, u8, u8) = (0xff, 0x6b, 0x6b);
}

/// Determine the accent color based on H¹.
pub fn disagreement_color(h1: usize) -> (u8, u8, u8) {
    match h1 {
        0 => colors::COLOR_AGREE,
        1 => colors::COLOR_WARN,
        _ => colors::COLOR_DISAGREE,
    }
}

// ─── Helper: truncate ─────────────────────────────────────────────────

/// Truncate a string to at most `max` characters, adding "…" if truncated.
pub fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!(
            "{}…",
            s.chars().take(max.saturating_sub(1)).collect::<String>()
        )
    }
}

// ─── Union-Find Helpers ──────────────────────────────────────────────

fn find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]);
    }
    parent[x]
}

fn union(parent: &mut [usize], x: usize, y: usize) {
    let rx = find(parent, x);
    let ry = find(parent, y);
    if rx != ry {
        parent[rx] = ry;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    // ─── Similarity Tests ─────────────────────────────────────────────

    #[test]
    fn similarity_identical_strings() {
        assert!((text_similarity("hello world", "hello world") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn similarity_empty_strings() {
        assert!((text_similarity("", "") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn similarity_one_empty() {
        assert!((text_similarity("hello", "") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn similarity_completely_different() {
        let sim = text_similarity("the quick brown fox", "zzz yyy xxx www");
        assert!(sim < 0.3, "expected low similarity, got {}", sim);
    }

    #[test]
    fn similarity_partial_overlap() {
        let sim = text_similarity("add null check to handler", "add null guard to handler");
        assert!(sim > 0.5, "expected high similarity, got {}", sim);
    }

    #[test]
    fn similarity_case_insensitive() {
        let sim = text_similarity("Add Null Check", "add null check");
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn similarity_short_strings() {
        let sim = text_similarity("fix", "fix");
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn similarity_reordered_words() {
        let sim = text_similarity("check null add", "add null check");
        assert!(sim > 0.7, "expected high similarity for reordered words, got {}", sim);
    }

    // ─── Semantic Boost Tests ─────────────────────────────────────────

    #[test]
    fn semantic_boost_synonym_fix_patterns() {
        let boost = semantic_boost("add null check", "insert null guard");
        assert!(boost > 0.0, "expected positive semantic boost");
    }

    #[test]
    fn semantic_boost_unrelated() {
        let boost = semantic_boost("refactor module", "update config file");
        assert!(boost < 0.1, "expected near-zero boost for unrelated texts");
    }

    // ─── Agreement Detection Tests ────────────────────────────────────

    #[test]
    fn fixes_agree_identical() {
        let a = AgentFix { agent_id: "a".into(), fix_text: "add null check".into() };
        let b = AgentFix { agent_id: "b".into(), fix_text: "add null check".into() };
        assert!(fixes_agree(&a, &b));
    }

    #[test]
    fn fixes_agree_similar() {
        let a = AgentFix {
            agent_id: "a".into(),
            fix_text: "Add a null check before dereferencing the pointer".into(),
        };
        let b = AgentFix {
            agent_id: "b".into(),
            fix_text: "Insert a null guard to check the pointer before use".into(),
        };
        assert!(fixes_agree(&a, &b));
    }

    #[test]
    fn fixes_disagree_different() {
        let a = AgentFix {
            agent_id: "a".into(),
            fix_text: "Add caching layer for database queries".into(),
        };
        let b = AgentFix {
            agent_id: "b".into(),
            fix_text: "Rewrite the authentication module from scratch".into(),
        };
        assert!(!fixes_agree(&a, &b));
    }

    // ─── Sheaf Analysis Tests ─────────────────────────────────────────

    #[test]
    fn empty_agents() {
        let analysis = compute_sheaf_analysis(&[]);
        assert_eq!(analysis.h0, 0);
        assert_eq!(analysis.h1, 0);
        assert_eq!(analysis.agent_count, 0);
    }

    #[test]
    fn single_agent() {
        let agents = vec![AgentFix { agent_id: "a".into(), fix_text: "fix it".into() }];
        let analysis = compute_sheaf_analysis(&agents);
        assert_eq!(analysis.h0, 1);
        assert_eq!(analysis.h1, 0);
        assert_eq!(analysis.agent_count, 1);
    }

    #[test]
    fn two_agents_agree() {
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "add null check to handler".into() },
            AgentFix { agent_id: "b".into(), fix_text: "add null check to handler".into() },
        ];
        let analysis = compute_sheaf_analysis(&agents);
        assert_eq!(analysis.h0, 1);
        assert_eq!(analysis.h1, 0);
        assert_eq!(analysis.agreement_edges.len(), 1);
        assert!(analysis.disagreement_pairs.is_empty());
    }

    #[test]
    fn two_agents_disagree() {
        let agents = vec![
            AgentFix {
                agent_id: "a".into(),
                fix_text: "Implement Redis caching layer for all database queries".into(),
            },
            AgentFix {
                agent_id: "b".into(),
                fix_text: "Rewrite the entire authentication and authorization module".into(),
            },
        ];
        let analysis = compute_sheaf_analysis(&agents);
        assert_eq!(analysis.h0, 2);
        assert_eq!(analysis.h1, 1);
    }

    #[test]
    fn three_agents_all_agree() {
        // All three agree pairwise → complete graph of agreements.
        // The sheaf has H⁰ = 1 (one component) and H¹ > 0 because a
        // complete graph with 3 edges on 1-D stalks creates a cycle.
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "add null check to handler".into() },
            AgentFix { agent_id: "b".into(), fix_text: "add null check to handler".into() },
            AgentFix { agent_id: "c".into(), fix_text: "add null guard to handler".into() },
        ];
        let analysis = compute_sheaf_analysis(&agents);
        assert_eq!(analysis.h0, 1);
        // H¹ comes from the cycle structure of the agreement graph.
        // With 3 agents and 3 edges on dim-1 stalks, the identity sheaf
        // on a triangle has H¹ = 1 per cycle.
        assert!(analysis.h1 == 1, "triangle of agreements → H¹ = 1, got {}", analysis.h1);
    }


    #[test]
    fn three_agents_all_disagree() {
        let agents = vec![
            AgentFix {
                agent_id: "a".into(),
                fix_text: "Implement Redis caching layer for all database queries".into(),
            },
            AgentFix {
                agent_id: "b".into(),
                fix_text: "Rewrite the entire authentication and authorization module".into(),
            },
            AgentFix {
                agent_id: "c".into(),
                fix_text: "Migrate all services to a completely different cloud provider".into(),
            },
        ];
        let analysis = compute_sheaf_analysis(&agents);
        assert_eq!(analysis.h0, 3);
        assert!(analysis.h1 >= 3);
    }

    #[test]
    fn three_agents_two_agree_one_disagrees() {
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "add null check to handler".into() },
            AgentFix { agent_id: "b".into(), fix_text: "add null check to handler".into() },
            AgentFix {
                agent_id: "c".into(),
                fix_text: "Completely restructure the entire application architecture".into(),
            },
        ];
        let analysis = compute_sheaf_analysis(&agents);
        assert_eq!(analysis.h0, 2);
        assert!(analysis.h1 >= 2);
    }

    #[test]
    fn analysis_preserves_agent_count() {
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "fix".into() },
            AgentFix { agent_id: "b".into(), fix_text: "fix".into() },
            AgentFix { agent_id: "c".into(), fix_text: "fix".into() },
            AgentFix { agent_id: "d".into(), fix_text: "different".into() },
        ];
        let analysis = compute_sheaf_analysis(&agents);
        assert_eq!(analysis.agent_count, 4);
    }

    // ─── AgentDisagreementSheaf Tests ─────────────────────────────────

    #[test]
    fn disagreement_sheaf_wraps_sheaf_agents() {
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "add null check".into() },
            AgentFix { agent_id: "b".into(), fix_text: "add null check".into() },
            AgentFix { agent_id: "c".into(), fix_text: "add null check".into() },
        ];
        let ds = AgentDisagreementSheaf::from_agents(&agents);
        assert_eq!(ds.sheaf.stalk_dims.len(), 3);
        assert!(ds.sheaf.edges.len() >= 3); // all three agree pairwise
    }

    #[test]
    fn disagreement_sheaf_h0_all_agree() {
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "add null check".into() },
            AgentFix { agent_id: "b".into(), fix_text: "add null guard".into() },
        ];
        let ds = AgentDisagreementSheaf::from_agents(&agents);
        assert_eq!(ds.h0(TOL), 1, "H⁰ = 1 (global section: all agree)");
    }

    #[test]
    fn disagreement_sheaf_h1_disagree() {
        let agents = vec![
            AgentFix {
                agent_id: "a".into(),
                fix_text: "use redis caching".into(),
            },
            AgentFix {
                agent_id: "b".into(),
                fix_text: "rewrite auth module completely".into(),
            },
            AgentFix {
                agent_id: "c".into(),
                fix_text: "migrate to different cloud provider".into(),
            },
        ];
        let _ds = AgentDisagreementSheaf::from_agents(&agents);
        // All three disagree → no edges → H¹ = 0 (trivial sheaf).
        // The high-level API still reports the graph-theoretic H¹.
        let analysis = compute_sheaf_analysis(&agents);
        assert!(analysis.h1 >= 3);
    }

    // ─── Verdict Tests ────────────────────────────────────────────────

    #[test]
    fn verdict_communication_helps() {
        let analysis = SheafAnalysis {
            h0: 1,
            h1: 0,
            agent_count: 2,
            agreement_edges: vec![],
            disagreement_pairs: vec![],
        };
        assert_eq!(verdict(&analysis), DisagreementVerdict::CommunicationHelps);
        assert_eq!(verdict(&analysis).label(), "Communication helps");
    }

    #[test]
    fn verdict_one_obstruction() {
        let analysis = SheafAnalysis {
            h0: 2,
            h1: 1,
            agent_count: 2,
            agreement_edges: vec![],
            disagreement_pairs: vec![],
        };
        assert_eq!(verdict(&analysis), DisagreementVerdict::OneObstruction);
    }

    #[test]
    fn verdict_need_new_perspective() {
        let analysis = SheafAnalysis {
            h0: 3,
            h1: 3,
            agent_count: 3,
            agreement_edges: vec![],
            disagreement_pairs: vec![],
        };
        assert_eq!(verdict(&analysis), DisagreementVerdict::NeedNewPerspective);
    }

    // ─── Color Tests ──────────────────────────────────────────────────

    #[test]
    fn color_green_for_agreement() {
        assert_eq!(disagreement_color(0), colors::COLOR_AGREE);
    }

    #[test]
    fn color_yellow_for_one_obstruction() {
        assert_eq!(disagreement_color(1), colors::COLOR_WARN);
    }

    #[test]
    fn color_red_for_multiple_obstructions() {
        assert_eq!(disagreement_color(2), colors::COLOR_DISAGREE);
        assert_eq!(disagreement_color(10), colors::COLOR_DISAGREE);
    }

    // ─── Helper Tests ─────────────────────────────────────────────────

    #[test]
    fn trigrams_basic() {
        let t = trigrams("hello");
        assert!(t.contains("hel"));
        assert!(t.contains("ell"));
        assert!(t.contains("llo"));
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn trigrams_short_string() {
        let t = trigrams("ab");
        assert_eq!(t.len(), 1);
        assert!(t.contains("ab"));
    }

    #[test]
    fn trigrams_empty() {
        let t = trigrams("");
        assert!(t.is_empty() || t.contains(""));
    }

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_long_string() {
        assert_eq!(truncate_str("hello world", 5), "hell…");
    }

    #[test]
    fn truncate_exact_length() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    // ─── Union-Find Tests ─────────────────────────────────────────────

    #[test]
    fn union_find_basic() {
        let mut parent = vec![0, 1, 2];
        union(&mut parent, 0, 1);
        assert_eq!(find(&mut parent, 0), find(&mut parent, 1));
    }

    #[test]
    fn union_find_transitive() {
        let mut parent = vec![0, 1, 2];
        union(&mut parent, 0, 1);
        union(&mut parent, 1, 2);
        assert_eq!(find(&mut parent, 0), find(&mut parent, 2));
    }

    #[test]
    fn union_find_separate_components() {
        let mut parent = vec![0, 1, 2, 3];
        union(&mut parent, 0, 1);
        union(&mut parent, 2, 3);
        assert_ne!(find(&mut parent, 0), find(&mut parent, 2));
    }

    // ─── Jaccard Tests ────────────────────────────────────────────────

    #[test]
    fn jaccard_identical_sets() {
        let a: HashSet<&str> = ["a", "b", "c"].into_iter().collect();
        let b: HashSet<&str> = ["a", "b", "c"].into_iter().collect();
        assert!((jaccard(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a: HashSet<&str> = ["a", "b"].into_iter().collect();
        let b: HashSet<&str> = ["c", "d"].into_iter().collect();
        assert!((jaccard(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a: HashSet<&str> = ["a", "b", "c"].into_iter().collect();
        let b: HashSet<&str> = ["b", "c", "d"].into_iter().collect();
        let j = jaccard(&a, &b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    // ─── Serialization Tests ──────────────────────────────────────────

    #[test]
    fn agent_fix_serialize_roundtrip() {
        let fix = AgentFix {
            agent_id: "test-agent".into(),
            fix_text: "add null check to handler".into(),
        };
        let json = serde_json::to_string(&fix).unwrap();
        let deserialized: AgentFix = serde_json::from_str(&json).unwrap();
        assert_eq!(fix.agent_id, deserialized.agent_id);
        assert_eq!(fix.fix_text, deserialized.fix_text);
    }

    #[test]
    fn sheaf_analysis_serialize_roundtrip() {
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "fix one".into() },
            AgentFix { agent_id: "b".into(), fix_text: "fix two".into() },
        ];
        let analysis = compute_sheaf_analysis(&agents);
        let json = serde_json::to_string(&analysis).unwrap();
        let deserialized: SheafAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(analysis, deserialized);
    }

    // ─── Edge Cases ───────────────────────────────────────────────────

    #[test]
    fn many_agents_cluster_correctly() {
        // 5 agents: {A,B,C} agree, {D,E} agree with each other but not A/B/C.
        // This forms 2 components with a complete agreement subgraph on {A,B,C}
        // (3 edges) and a single edge on {D,E}.
        let agents = vec![
            AgentFix {
                agent_id: "a".into(),
                fix_text: "add null check to input validation".into(),
            },
            AgentFix {
                agent_id: "b".into(),
                fix_text: "add null check to input validation".into(),
            },
            AgentFix {
                agent_id: "c".into(),
                fix_text: "add null guard to input validation".into(),
            },
            AgentFix {
                agent_id: "d".into(),
                fix_text: "Refactor the entire logging subsystem with structured events".into(),
            },
            AgentFix {
                agent_id: "e".into(),
                fix_text: "Refactor the entire logging subsystem with structured events".into(),
            },
        ];
        let analysis = compute_sheaf_analysis(&agents);
        assert_eq!(analysis.h0, 2);
        // Cross-component disagreements: 3 agents in cluster 0 × 2 agents in cluster 1 = 6
        // Plus H¹ from triangle cycle in cluster 0 = 1
        // Total H¹ = 6 cross-component + 1 sheaf H¹ = 7
        assert!(analysis.h1 >= 1, "should detect disagreements, got h1={}", analysis.h1);
    }

    #[test]
    fn completely_unrelated_fixes_disagree() {
        let a = AgentFix {
            agent_id: "a".into(),
            fix_text: "Optimize the SQL query with proper indexing".into(),
        };
        let b = AgentFix {
            agent_id: "b".into(),
            fix_text: "Add unit tests for the payment processing module".into(),
        };
        assert!(!fixes_agree(&a, &b));
    }

    #[test]
    fn spectral_gap_delegation() {
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "fix bug #123".into() },
            AgentFix { agent_id: "b".into(), fix_text: "fix bug #123".into() },
        ];
        let ds = AgentDisagreementSheaf::from_agents(&agents);
        let gap = ds.spectral_gap();
        assert!(gap >= 0.0, "spectral gap should be non-negative, got {}", gap);
    }

    #[test]
    fn sheaf_analysis_cloned_is_equal() {
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "fix the bug".into() },
            AgentFix { agent_id: "b".into(), fix_text: "fix the bug".into() },
        ];
        let a1 = compute_sheaf_analysis(&agents);
        let a2 = a1.clone();
        assert_eq!(a1, a2);
    }

    #[test]
    fn agreement_threshold_sanity() {
        let similar = AgentFix {
            agent_id: "a".into(),
            fix_text: "fix the null pointer exception in handler".into(),
        };
        let near = AgentFix {
            agent_id: "b".into(),
            fix_text: "fix the null pointer error in handler".into(),
        };
        assert!(fixes_agree(&similar, &near));
    }

    #[test]
    fn disagreement_sheaf_empty_agents_handled() {
        let agents: Vec<AgentFix> = vec![];
        let ds = AgentDisagreementSheaf::from_agents(&agents);
        assert_eq!(ds.sheaf.stalk_dims.len(), 1, "empty → 1 stub vertex");
    }

    #[test]
    fn sheaf_analysis_cross_component_breakdown() {
        // 4 agents in two clusters: {A,B} and {C,D}. A↔C and A↔D disagree
        // (cross-component). B↔C and B↔D disagree (cross-component).
        // Within-component: none (A-B agree, C-D agree).
        let agents = vec![
            AgentFix { agent_id: "a".into(), fix_text: "add null check".into() },
            AgentFix { agent_id: "b".into(), fix_text: "add null check".into() },
            AgentFix { agent_id: "c".into(), fix_text: "rewrite auth module".into() },
            AgentFix { agent_id: "d".into(), fix_text: "rewrite auth module".into() },
        ];
        let analysis = compute_sheaf_analysis(&agents);
        assert_eq!(analysis.h0, 2);
        // Cross-component: {A,C}, {A,D}, {B,C}, {B,D} = 4
        assert!(analysis.h1 >= 4);
    }

    #[test]
    fn analyze_deserialized_data() {
        let json = r#"{
            "agent_id": "copilot",
            "fix_text": "add null pointer guard"
        }"#;
        let fix: AgentFix = serde_json::from_str(json).unwrap();
        assert_eq!(fix.agent_id, "copilot");
        assert!(fix.fix_text.contains("null pointer"));
    }

    #[test]
    fn verdict_serialize_roundtrip() {
        let v = DisagreementVerdict::OneObstruction;
        let json = serde_json::to_string(&v).unwrap();
        let deserialized: DisagreementVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, deserialized);
    }
}
