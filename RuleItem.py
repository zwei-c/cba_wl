import pandas as pd


class RuleItem():
    def __init__(self):
        self.conditions: dict = {}
        self.class_: str = ""
        self.support: float = 0.0
        self.confidence: float = 0.0
        self.lift: float = 0.0
        self.is_ruleitem = False

    def set_ruleitem(self, item: dict, data: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.1):
        # Declare Instance Variables
        support: float = 0.0
        condition_support: float = 0.0
        confidence: float = 0.0
        # Definite Instance Variables
        conditions = item["conditions"]
        class_ = item["class"]
        data_frame_conditions = None

        for key in conditions:
            if (data_frame_conditions is None):
                data_frame_conditions = data[key] == conditions[key]
            else:
                data_frame_conditions &= data[key] == conditions[key]
        data_frame_conditions_with_class = data_frame_conditions & (
            data["class"] == class_)

        support = data[data_frame_conditions_with_class].shape[0] / \
            data.shape[0]
        condition_support = data[data_frame_conditions].shape[0] / \
            data.shape[0]
        if support >= min_support:
            confidence = support / condition_support
            if confidence >= min_confidence:
                self.conditions = conditions
                self.class_ = class_
                self.support = support
                self.confidence = confidence
                self.is_ruleitem = True
                # self.lift = confidence / (data[data["class"] == class_].shape[0] / data.shape[0])
                del data
                return True
            else:
                del data
                return False
        else:
            del data
            return False

    def print_ruleitem(self):
        if not self.is_ruleitem:
            print("This is not a ruleitem")
            return
        string = ""
        for key, value in self.conditions.items():
            string += f"{key}={value} "
        string += f"=> class={self.class_} "
        string += f"(support={self.support}, confidence={self.confidence}, lift={self.lift})"
        print(string)
