from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

if __name__ == "__main__":
    major_ver, minor_ver, _ = version.parse(tb.__version__).release
    assert major_ver >= 2 and minor_ver >= 3, \
        "This notebook requires TensorBoard 2.3 or later."
    print("TensorBoard version: ", tb.__version__)
    
    experiment_id = "xF9ygK9zReOfvAyMNTe6VA"
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df
    
    print(df["run"].unique())
    print(df["tag"].unique())
    
    dfw = experiment.get_scalars(pivot=False)
    dfw
    
    csv_path = '/tmp/tb_experiment_1.csv'
    dfw.to_csv(csv_path, index=False)
    dfw_roundtrip = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(dfw_roundtrip, dfw)
    
    # Filter the DataFrame to only validation data, which is what the subsequent
    # analyses and visualization will be focused on.
    dfw_validation = dfw[dfw.run.str.endswith("/validation")]
    # Get the optimizer value for each row of the validation DataFrame.
    optimizer_validation = dfw_validation.run.apply(lambda run: run.split(",")[0])

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=dfw_validation, x="step", y="epoch_accuracy",
                 hue=optimizer_validation).set_title("accuracy")
    plt.subplot(1, 2, 2)
    sns.lineplot(data=dfw_validation, x="step", y="epoch_loss",
                 hue=optimizer_validation).set_title("loss")




