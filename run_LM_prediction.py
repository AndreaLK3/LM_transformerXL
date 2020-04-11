import TW
import InputFacilities.Input as Input
import daemon

### The dataset_valuename is the dataset name in string format. For instance it can be "Danish", or "Spanish"
def main(dataset_valuename):

    dataset_valuename = dataset_valuename.upper()

    dataset = Input.Dataset[dataset_valuename]

    LM = TW.LM_TransformerXL(dataset, flag_text_or_manual=False, input_filepath=".", flag_verbose=True) #verbose for now

    while True:
        context_touse_forprediction = input()
        LM.predict(context_touse_forprediction)


def start_background_service():
    pass