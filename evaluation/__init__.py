"""Architecture comparison harness.

The production agent reaches every tool through three meta tools
(search_tools / get_tool / run_tools). This package builds the control
group: the same agent, same model, same middlewares — but with every tool
schema wired directly into the model. Run one task against each and compare
the usage footers. See evaluation/README.md.
"""
