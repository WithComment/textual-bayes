import textgrad as tg
from textgrad.engine import EngineLM
from textgrad.variable import Variable


class LikelihoodLoss(tg.loss.MultiFieldEvaluation):

    def __init__(
        self,
        evaluation_instruction: Variable | str,
        role_description: str,
        input_roles: list[str],
        engine: EngineLM | str = None,
        system_prompt: Variable = None,
    ):
        if isinstance(evaluation_instruction, str):
            evaluation_instruction = tg.Variable(
                evaluation_instruction,
                requires_grad=False,
                role_description=role_description,
            )

        super().__init__(
            evaluation_instruction=evaluation_instruction,
            role_descriptions=input_roles,
            engine=engine,
            system_prompt=system_prompt,
        )
