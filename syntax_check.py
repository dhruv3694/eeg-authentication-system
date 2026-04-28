import ast
with open("auth_server.py") as f:
    src = f.read()
ast.parse(src)
print("auth_server.py: Syntax OK")
