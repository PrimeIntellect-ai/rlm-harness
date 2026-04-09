---
name: edit
description: Replace a unique string in a file. old_str must appear exactly once in the file.
---

### edit
Replace a unique string in a file. old_str must appear exactly once.
  edit(path="src/auth.py", old_str="return self._token", new_str="return self._generate_token()")
