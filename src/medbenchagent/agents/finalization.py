"""v2 delivery finalization boundary."""
def compile_finalization_agent(*, checkpointer=None):
    raise ValueError("Legacy finalization Agent removed; v2 pipeline publishes audited delivery")

def restore_delivery(context):
    return {"delivery_path": context.get("delivery_path"), "delivery_counts": context.get("delivery_counts", {})}
