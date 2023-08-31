import pandas as pd


class Cba():
    def __init__(self, data: pd.DataFrame, rules: list):
        self.data = data
        self.rules = rules
        self.final_rules = []
        self.default = str(self.data['class'].mode()[0])

    def cover(self):
        data = self.data
        for rule in self.rules:
            if data.shape[0] == 0:
                break
            conditions = rule.conditions
            data_frame_conditions = None

            for key in conditions:
                if (data_frame_conditions is None):
                    data_frame_conditions = data[key] == conditions[key]
                else:
                    data_frame_conditions &= data[key] == conditions[key]

            if data[data_frame_conditions].shape[0] > 0:
                data = data[~data_frame_conditions]
                if data.shape[0] != 0:
                    self.default = str(data['class'].mode()[0])
                self.final_rules.append({'rule': rule, 'error': 0})
                self.compare()
            else:
                continue
        # 找到錯誤數最少的規則 刪除他之後的規則
        error = 1000000000
        for index in range(len(self.final_rules)):
            if self.final_rules[index]['error'] < error:
                error = self.final_rules[index]['error']
                error_index = index
        self.final_rules = self.final_rules[:error_index]

    def compare(self):

        def compare_sub(item, rule):
            conditions = rule.conditions
            for key in conditions:
                if item[key] != conditions[key]:
                    return False
            return True

        def get_errors(data, test_data):
            error = 0
            for index in range(data.shape[0]):
                if data.loc[index, 'class'] != test_data.loc[index, 'class']:
                    error += 1
            return error

        test_data = self.data.drop(columns=['class'], axis=1)

        for item in test_data.iterrows():
            for rule in self.final_rules:
                if compare_sub(item[1], rule['rule']):
                    test_data.loc[item[0], 'class'] = rule['rule'].class_
                    break
                else:
                    test_data.loc[item[0], 'class'] = self.default

        error = get_errors(self.data, test_data)
        self.final_rules[-1]['error'] = error
