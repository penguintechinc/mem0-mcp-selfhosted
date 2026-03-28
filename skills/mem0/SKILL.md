---
name: mem0
description: Use when reading or writing persistent cross-session memories via the mem0 MCP server — searching prior context, storing learnings, filtering by author/project/branch, or managing existing memories.
---

# mem0 Persistent Memory

Semantic vector memory backed by Qdrant + Ollama (nomic-embed-text). Default scope: `user_id=justinb`.

## When to Search vs Add

**Search at session start** — before asking the user to re-explain context:
```
search_memories("project preferences coding style")
search_memories("<repo or topic name>")
```

**Add after discovering** something worth keeping across sessions: preferences, decisions, patterns, tool configs, gotchas. Write as a complete self-contained statement — include the *why*.

**Update, don't delete+re-add** — preserves the vector embedding and memory ID:
```
update_memory(memory_id="<id>", text="updated fact here")
```

## Tool Quick Reference

| Tool | Use for |
|------|---------|
| `search_memories(query)` | Semantic search — use natural language |
| `add_memory(text)` | Store new fact/preference/decision |
| `get_memories()` | List ALL memories (no query needed) |
| `get_memory(memory_id)` | Fetch one memory by ID |
| `update_memory(memory_id, text)` | Overwrite existing memory |
| `delete_memory(memory_id)` | Remove a single memory |

## Metadata Schema

Every memory is auto-stamped with:

```json
{
  "user":   "justinb",       // MEM0_AUTHOR_ID (falls back to MEM0_USER_ID)
  "team":   "infosec",       // MEM0_TEAM_ID
  "repo":   "https://github.com/org/repo",  // auto-detected from CWD (omitted if not a git repo)
  "branch": "main"           // current git branch (omitted if not a git repo or detached HEAD)
}
```

Caller-supplied `metadata={}` wins on collision.

## Filtering Search Results

Use Qdrant metadata filter syntax in the `filters` parameter — **not** a bare `user_id` argument:

```python
# By author
search_memories("auth patterns", filters={"user": {"eq": "justinb"}})

# By repo
search_memories("deploy config", filters={"repo": {"eq": "https://github.com/org/my-project"}})

# By branch
search_memories("WIP notes", filters={"branch": {"eq": "feature/auth-rewrite"}})

# By team
search_memories("shared conventions", filters={"team": {"eq": "infosec"}})
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `user_id="justinb"` as filter arg | Use `filters={"user": {"eq": "justinb"}}` |
| Delete + re-add to update | Use `update_memory(memory_id, text)` |
| Storing vague text | Write complete statements: "Justin prefers X because Y" |
| Searching with keyword only | Use natural language — it's vector similarity, not grep |
| `add_memory` returns `{"results": []}` silently | LLM extracted nothing (small models + long text). Use `infer=False` to store raw text directly |

## ⚠️ Risk: Vector Dimension Mismatch

If you see **"Vector dimension error: expected dim: X, got Y"**, the Qdrant collection was created with the wrong dims. This happens when `MEM0_EMBED_MODEL` or `MEM0_EMBED_DIMS` changes between sessions, or when multiple projects share the same Qdrant instance with different configs.

**`MEM0_EMBED_DIMS` must match the model's actual output:** `bge-m3` → 1024, `nomic-embed-text` → 768.

Fix: delete the stale collection and let it be recreated:
```bash
curl -X DELETE http://localhost:6333/collections/mem0_mcp_selfhosted
```
Then retry — the server recreates the collection at the configured dims on the next write.
