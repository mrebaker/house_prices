"""
Tools for tracking and predicting UK residential property values, based on data from Land Registry.
"""


class Prop:
    """
    A residential property in the UK, based on info both from Land Registry and privately held.
    """
    def __init__(self, prop_type, address):
        self.prop_type = prop_type
        self.address = address
        self.value_history = self.get_value_history()
        self.predicted_value = 0

    def get_value_history(self):
        """
        Get the history of known values for this property
        :return:
        """
        # TODO: either look up values on Land Registry, or search the database
        return []

    def predict_value(self):
        """
        Predicts the current value of a property based on the full Land Registry dataset.
        :param prop:
        :return:
        """

        # TODO: implement predictive model and return predicted value instead
        self.predicted_value = 0


def load_initial_data():
    """
    Takes Land Registry price data from a text file and creates a database from it.
    :return:
    """


def load_new_data():
    """
    Takes Land Registry price data for a single month and updates the database with it.
    :return:
    """


def validate_new_data():
    """
    Checks that the data to be uploaded does not already existing in the database.
    :return:
    """





if __name__ == '__main__':
    pass
