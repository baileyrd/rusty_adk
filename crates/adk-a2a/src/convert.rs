//! Translation between the ADK and A2A content models.
//!
//! The two are close but not identical, and every place they differ is a
//! decision rather than a mechanical mapping. Those decisions live here so
//! they can be read — and tested — in one place.

use adk_core::{Blob, Content, FileData, Part as AdkPart, Role as AdkRole};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rusty_a2a::types::{Message as A2aMessage, Part as A2aPart, PartContent, Role as A2aRole};

/// Converts an inbound A2A message into the ADK content an agent reads.
///
/// A2A's `data` parts have no ADK counterpart — ADK carries structured values
/// on `Event.output` and in tool payloads, not inside conversation content —
/// so they arrive as their compact JSON text. That keeps the information
/// rather than dropping the part, and a model reads it fine.
pub fn message_to_content(message: &A2aMessage) -> Content {
    let role = match message.role {
        A2aRole::Agent => AdkRole::Model,
        // `ROLE_UNSPECIFIED` on an inbound message is treated as the user
        // speaking: it is the only role a client can legitimately be in.
        A2aRole::User | A2aRole::Unspecified => AdkRole::User,
    };
    Content::new(role, message.parts.iter().map(part_to_adk).collect())
}

/// Converts one A2A part into its ADK equivalent.
pub fn part_to_adk(part: &A2aPart) -> AdkPart {
    let mime = |fallback: &str| {
        part.media_type
            .clone()
            .unwrap_or_else(|| fallback.to_string())
    };
    match &part.content {
        PartContent::Text { text } => AdkPart::Text(text.clone()),
        PartContent::Raw { raw } => AdkPart::InlineData(Blob {
            mime_type: mime("application/octet-stream"),
            // ADK holds inline bytes already base64-encoded, matching the
            // `google.genai` wire format; A2A hands them over decoded.
            data: STANDARD.encode(raw),
        }),
        PartContent::Url { url } => AdkPart::FileData(FileData {
            mime_type: mime("application/octet-stream"),
            file_uri: url.clone(),
        }),
        PartContent::Data { data } => AdkPart::Text(data.to_string()),
    }
}

/// Converts one ADK part into its A2A equivalent.
///
/// Returns `None` for parts that are internal to a run rather than content a
/// caller should receive: reasoning traces, and the function call/response
/// pair that drives ADK's tool loop. A2A has no vocabulary for either, and
/// leaking a model's `Thought` to a remote peer would be a privacy surprise.
pub fn part_to_a2a(part: &AdkPart) -> Option<A2aPart> {
    match part {
        AdkPart::Text(text) => Some(A2aPart::text(text)),
        AdkPart::InlineData(blob) => {
            // ADK stores base64; A2A wants the bytes, and re-encodes them
            // itself on the wire. A payload ADK could not decode is passed
            // through as text rather than silently dropped.
            match STANDARD.decode(blob.data.as_bytes()) {
                Ok(bytes) => Some(A2aPart::raw(bytes).with_media_type(&blob.mime_type)),
                Err(_) => Some(A2aPart::text(&blob.data).with_media_type(&blob.mime_type)),
            }
        }
        AdkPart::FileData(file) => {
            Some(A2aPart::url(&file.file_uri).with_media_type(&file.mime_type))
        }
        AdkPart::Thought(_) | AdkPart::FunctionCall(_) | AdkPart::FunctionResponse(_) => None,
    }
}

/// Builds an A2A agent message from ADK content, dropping internal parts.
///
/// Returns `None` when nothing survives the filter — an ADK event carrying
/// only a function call has nothing to say to an A2A caller.
pub fn content_to_message(content: &Content) -> Option<A2aMessage> {
    let parts: Vec<A2aPart> = content.parts.iter().filter_map(part_to_a2a).collect();
    if parts.is_empty() {
        return None;
    }
    Some(A2aMessage::new(A2aRole::Agent, parts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_survives_both_directions() {
        let a2a = A2aMessage::user_text("hello");
        let content = message_to_content(&a2a);
        assert_eq!(content.role, AdkRole::User);
        assert_eq!(content.text(), "hello");

        let back = content_to_message(&Content::model_text("hi")).unwrap();
        assert_eq!(back.role, A2aRole::Agent);
        assert_eq!(back.text(), "hi");
    }

    #[test]
    fn agent_role_maps_to_model() {
        let a2a = A2aMessage::agent_text("from the agent");
        assert_eq!(message_to_content(&a2a).role, AdkRole::Model);
    }

    #[test]
    fn an_unspecified_role_is_read_as_the_user_speaking() {
        let message = A2aMessage::new(A2aRole::Unspecified, vec![A2aPart::text("hi")]);
        assert_eq!(message_to_content(&message).role, AdkRole::User);
    }

    #[test]
    fn raw_bytes_become_base64_inline_data_and_back() {
        let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let part = A2aPart::raw(bytes.clone()).with_media_type("image/png");

        let adk = part_to_adk(&part);
        let AdkPart::InlineData(blob) = &adk else {
            panic!("expected inline data, got {adk:?}");
        };
        assert_eq!(blob.mime_type, "image/png");
        assert_eq!(STANDARD.decode(&blob.data).unwrap(), bytes);

        // ...and the round trip returns the original bytes, not the base64.
        let back = part_to_a2a(&adk).unwrap();
        assert_eq!(back.media_type.as_deref(), Some("image/png"));
        assert!(matches!(&back.content, PartContent::Raw { raw } if raw == &bytes));
    }

    #[test]
    fn raw_bytes_without_a_media_type_get_a_generic_one() {
        let adk = part_to_adk(&A2aPart::raw(vec![1, 2, 3]));
        let AdkPart::InlineData(blob) = &adk else {
            panic!("expected inline data");
        };
        assert_eq!(blob.mime_type, "application/octet-stream");
    }

    #[test]
    fn urls_become_file_references_and_back() {
        let part = A2aPart::url("https://example.com/f.pdf").with_media_type("application/pdf");
        let adk = part_to_adk(&part);
        let AdkPart::FileData(file) = &adk else {
            panic!("expected file data, got {adk:?}");
        };
        assert_eq!(file.file_uri, "https://example.com/f.pdf");

        let back = part_to_a2a(&adk).unwrap();
        assert!(matches!(&back.content, PartContent::Url { url } if url == &file.file_uri));
        assert_eq!(back.media_type.as_deref(), Some("application/pdf"));
    }

    #[test]
    fn structured_data_arrives_as_its_json_text() {
        let adk = part_to_adk(&A2aPart::data(json!({"city": "Kyoto"})));
        assert_eq!(adk.as_text(), Some(r#"{"city":"Kyoto"}"#));
    }

    #[test]
    fn internal_parts_are_not_exposed_to_a_peer() {
        use adk_core::{FunctionCall, FunctionResponse};

        assert!(part_to_a2a(&AdkPart::Thought("reasoning".into())).is_none());
        assert!(part_to_a2a(&AdkPart::FunctionCall(FunctionCall::new(
            "t",
            Default::default()
        )))
        .is_none());
        assert!(part_to_a2a(&AdkPart::FunctionResponse(FunctionResponse {
            id: None,
            name: "t".into(),
            response: json!({}),
        }))
        .is_none());
    }

    #[test]
    fn content_with_nothing_public_produces_no_message() {
        let content = Content::new(AdkRole::Model, vec![AdkPart::Thought("thinking".into())]);
        assert!(content_to_message(&content).is_none());
    }

    #[test]
    fn undecodable_inline_data_is_preserved_as_text() {
        let content = AdkPart::InlineData(Blob {
            mime_type: "text/plain".into(),
            data: "not base64!!".into(),
        });
        let part = part_to_a2a(&content).unwrap();
        assert!(matches!(&part.content, PartContent::Text { text } if text == "not base64!!"));
    }
}
