//! The `skill` tool: serve the full text of a skill that startup discovery
//! indexed.
//!
//! Personal skills live under `~/.omega-system`, which the filesystem tools'
//! `protected_home` shield blocks. This tool is the deliberate, narrow hole:
//! it reads only the canonical paths discovery recorded, selected by indexed
//! name, so the shield itself stays intact. A skill body is untrusted text
//! entering as a tool result, like `web_fetch` content.

use super::ToolDef;
use crate::instructions::Skill;

/// The registered tool name; the agent's tool policy lists it only when a
/// tool by this name is present.
pub const NAME: &str = "skill";

pub struct SkillTool {
    skills: Vec<Skill>,
}

impl SkillTool {
    pub fn new(skills: Vec<Skill>) -> Self {
        Self { skills }
    }

    /// Fail closed: exactly one field, naming an indexed skill.
    fn lookup(&self, input: &serde_json::Value) -> Result<&Skill, String> {
        let fields = input.as_object().ok_or("input must be an object")?;
        if let Some(field) = fields.keys().find(|key| *key != "name") {
            return Err(format!("unknown field: {field}"));
        }
        let name = fields
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or("missing required field: name")?;
        self.skills
            .iter()
            .find(|skill| skill.name == name)
            .ok_or_else(|| format!("unknown skill: {name}"))
    }
}

impl ToolDef for SkillTool {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        "Fetch the full text of a skill listed in the system prompt's Skills section. \
         Call it before performing a task that matches the skill's description."
    }

    fn input_schema(&self) -> serde_json::Value {
        let names: Vec<&str> = self.skills.iter().map(|s| s.name.as_str()).collect();
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": names,
                    "description": "Name of the skill, exactly as indexed"
                }
            },
            "required": ["name"],
            "additionalProperties": false
        })
    }

    fn validate(&self, input: &serde_json::Value) -> Result<(), String> {
        self.lookup(input).map(|_| ())
    }

    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        input["name"]
            .as_str()
            .map(|name| format!("loading skill {name}"))
    }

    fn run(
        &self,
        input: serde_json::Value,
        _out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        let skill = self.lookup(&input)?;
        skill
            .body()
            .map_err(|e| format!("cannot load skill {}: {e}", skill.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::discover;
    use serde_json::json;
    use std::fs;

    /// Skills come from real discovery over a tempdir, so the tool is tested
    /// against exactly the paths and bodies startup would hand it.
    struct Library {
        home: tempfile::TempDir,
        project: tempfile::TempDir,
    }

    impl Library {
        fn new(skills: &[(&str, &str)]) -> Self {
            let library = Self {
                home: tempfile::tempdir().unwrap(),
                project: tempfile::tempdir().unwrap(),
            };
            fs::create_dir_all(library.dir()).unwrap();
            for (name, body) in skills {
                fs::write(library.dir().join(format!("{name}.md")), body).unwrap();
            }
            library
        }

        fn dir(&self) -> std::path::PathBuf {
            self.home.path().join("skills")
        }

        fn tool(&self) -> SkillTool {
            let discovered = discover(
                Some(self.home.path()),
                self.project.path(),
                false,
                &mut &b""[..],
                &mut Vec::new(),
            )
            .unwrap();
            SkillTool::new(discovered.skills().to_vec())
        }
    }

    #[test]
    fn metadata_is_free_read_only_and_lists_indexed_names() {
        let library = Library::new(&[("deploy", "Ship it."), ("review", "Check it.")]);
        let tool = library.tool();
        assert_eq!(tool.name(), "skill");
        assert!(tool.description().contains("Skills section"));
        assert_eq!(tool.cost(), 0);
        assert!(!tool.requires_confirmation());
        assert!(!tool.side_effecting(&json!({"name": "deploy"})));
        assert_eq!(
            tool.input_schema(),
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ["deploy", "review"],
                        "description": "Name of the skill, exactly as indexed"
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            })
        );
        assert_eq!(
            tool.format_status(&json!({"name": "deploy"})).as_deref(),
            Some("loading skill deploy")
        );
        assert!(tool.format_status(&json!({})).is_none());
    }

    #[test]
    fn happy_path_returns_the_whole_body() {
        let body = "# Deploy\n\nShip it.\n\n1. build\n2. push\n";
        let library = Library::new(&[("deploy", body)]);
        let tool = library.tool();
        let input = json!({"name": "deploy"});
        assert!(tool.validate(&input).is_ok());
        assert_eq!(tool.run(input, &mut Vec::new()).unwrap(), body);
    }

    #[test]
    fn validation_fails_closed_on_every_malformed_input() {
        let library = Library::new(&[("deploy", "Ship it.")]);
        let tool = library.tool();
        for (input, error) in [
            (json!("deploy"), "input must be an object"),
            (json!({}), "missing required field: name"),
            (json!({"name": 7}), "missing required field: name"),
            (
                json!({"name": "deploy", "path": "/etc/passwd"}),
                "unknown field: path",
            ),
            (json!({"name": "../deploy"}), "unknown skill: ../deploy"),
            (json!({"name": "absent"}), "unknown skill: absent"),
        ] {
            assert_eq!(tool.validate(&input).unwrap_err(), error);
            assert_eq!(tool.run(input, &mut Vec::new()).unwrap_err(), error);
        }
    }

    #[test]
    fn a_skill_deleted_after_discovery_is_a_clean_tool_error() {
        let library = Library::new(&[("deploy", "Ship it.")]);
        let tool = library.tool();
        fs::remove_file(library.dir().join("deploy.md")).unwrap();
        let error = tool
            .run(json!({"name": "deploy"}), &mut Vec::new())
            .unwrap_err();
        assert!(error.starts_with("cannot load skill deploy: cannot open"));
        assert!(error.ends_with("deploy.md: no longer exists"));
    }

    #[test]
    fn a_body_grown_past_the_cap_fails_whole_naming_the_skill() {
        let library = Library::new(&[("deploy", "Ship it.")]);
        let tool = library.tool();
        fs::write(library.dir().join("deploy.md"), vec![b'x'; 32 * 1024 + 1]).unwrap();
        let error = tool
            .run(json!({"name": "deploy"}), &mut Vec::new())
            .unwrap_err();
        assert!(error.starts_with("cannot load skill deploy: "));
        assert!(error.ends_with("deploy.md exceeds the 32768-byte limit"));
    }

    #[test]
    fn a_skill_swapped_for_a_symlink_after_discovery_is_not_followed() {
        let library = Library::new(&[("deploy", "Ship it.")]);
        let tool = library.tool();
        let secret = library.home.path().join(".env");
        fs::write(&secret, "API_KEY=secret").unwrap();
        let path = library.dir().join("deploy.md");
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&secret, &path).unwrap();
        let error = tool
            .run(json!({"name": "deploy"}), &mut Vec::new())
            .unwrap_err();
        assert!(error.contains("cannot open"));
        assert!(!error.contains("secret"));
    }
}
