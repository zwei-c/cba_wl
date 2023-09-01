from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")


class Validation():
    def __init__(self, data, rules, default):
        self.data = data
        self.rules = rules
        self.default = default
        self.answer = list(self.data['class'])

        self.forecast_result = self.predict()
        self.report = self.get_classification_report()

    def predict(self):
        forecast_result = self.data.drop(columns=['class'], axis=1)
        for index in forecast_result.index:
            check = None
            for rule in self.rules:
                check = True
                conditions = rule.conditions
                for condition in conditions:
                    if forecast_result.loc[index, condition] != conditions[condition]:
                        check = False
                        break
                if check:
                    forecast_result.loc[index, 'class'] = str(rule.class_)
                    break
            if not check:
                forecast_result.loc[index, 'class'] = str(self.default)
        return list(forecast_result['class'])

    def get_classification_report(self):
        y_test = [str(i) for i in self.answer]
        y_pred = [str(i) for i in self.forecast_result]
        print(classification_report(y_test, y_pred))
