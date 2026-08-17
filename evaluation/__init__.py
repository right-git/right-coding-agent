"""Architecture comparison harness.

The production agent reaches every tool through one meta tool (run_tools,
with in-script search_tools/get_tool discovery). This package builds the control
group: the same agent, same model, same middlewares — but with every tool
schema wired directly into the model. Run one task against each and compare
the usage footers. See evaluation/README.md.
"""
