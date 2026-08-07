//! Building an A2A `AgentCard` from an ADK agent.

use adk_agents::SharedAgent;
use rusty_a2a::types::{AgentCard, AgentInterface, AgentSkill};

/// Describes an ADK agent as an A2A [`AgentCard`].
///
/// An agent card is a manifest a peer reads before talking to an agent, and
/// ADK already carries most of what goes in one: the agent's name, what it is
/// for, and the sub-agents it can delegate to. Those sub-agents become skills,
/// since a skill is exactly A2A's notion of "a distinct thing this agent can
/// do" and delegation is how an ADK agent advertises the same.
///
/// The result is a starting point, not a finished card. Declare security
/// schemes, a provider, and richer skill metadata on the returned value before
/// serving it.
pub fn card_for_agent(
    agent: &SharedAgent,
    version: impl Into<String>,
    interface: AgentInterface,
) -> AgentCard {
    let description = if agent.description().is_empty() {
        format!("An ADK agent named {}.", agent.name())
    } else {
        agent.description().to_string()
    };

    let mut card = AgentCard::new(agent.name(), description, version, interface)
        // The bridge always drives the run through to a terminal status and
        // reports progress as it goes, so streaming is genuinely supported.
        .with_streaming(true);

    for sub in agent.sub_agents() {
        let sub_description = if sub.description().is_empty() {
            format!("Delegates to the {} agent.", sub.name())
        } else {
            sub.description().to_string()
        };
        card = card.with_skill(AgentSkill::new(sub.name(), sub.name(), sub_description));
    }
    card
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::LlmAgent;
    use adk_models::MockModel;
    use std::sync::Arc;

    fn agent(name: &str, description: &str) -> LlmAgent {
        LlmAgent::builder(name)
            .model(Arc::new(MockModel::new()))
            .description(description)
            .build()
            .unwrap()
    }

    #[test]
    fn the_card_carries_the_agents_own_description() {
        let card = card_for_agent(
            &agent("router", "Routes billing and technical questions.").shared(),
            "1.2.3",
            AgentInterface::json_rpc("http://localhost:8080"),
        );
        assert_eq!(card.name, "router");
        assert_eq!(card.description, "Routes billing and technical questions.");
        assert_eq!(card.version, "1.2.3");
        assert_eq!(card.capabilities.streaming, Some(true));
    }

    #[test]
    fn a_description_less_agent_still_gets_one() {
        let card = card_for_agent(
            &agent("plain", "").shared(),
            "0.1.0",
            AgentInterface::json_rpc("http://localhost:8080"),
        );
        assert_eq!(card.description, "An ADK agent named plain.");
    }

    #[test]
    fn sub_agents_become_skills() {
        let root = LlmAgent::builder("support")
            .model(Arc::new(MockModel::new()))
            .description("Front desk.")
            .sub_agent(agent("billing", "Handles refunds.").shared())
            .sub_agent(agent("technical", "Opens tickets.").shared())
            .build()
            .unwrap()
            .shared();

        let card = card_for_agent(
            &root,
            "0.1.0",
            AgentInterface::json_rpc("http://localhost:8080"),
        );
        let names: Vec<&str> = card.skills.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(names, vec!["billing", "technical"]);
        assert_eq!(card.skills[0].description, "Handles refunds.");
    }
}
