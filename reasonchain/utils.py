def dynamic_complexity_evaluator(step):
    """
    Dynamically evaluate the complexity of a step.
    :param step: Step description.
    :return: Complexity level ('low', 'medium', 'high').
    """
    if "fetch" in step.lower() or "retrieve" in step.lower():
        return "medium"
    elif "generate" in step.lower() or "evaluate" in step.lower():
        return "high"
    else:
        return "low"
