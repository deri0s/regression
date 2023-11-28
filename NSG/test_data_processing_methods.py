import numpy as np
import data_processing_methods as dpm
import pandas as pd

def test_process_ISRA():
    """
    Description
    -----------
        Test that we can replace the high and low values of ISRA
            scanner.
    """

    # Load data frame
    df = pd.read_excel(io='Fault Scanner Data 20190801 to 20210627.xlsx',
                       sheet_name='ISRA')

    # Date range we want to look at
    date1 = np.datetime64('2020-07-25')
    date2 = np.datetime64('2021-07-30')

    # Process ISRA data by replacing offending module data
    ff_processed, high_ff_locations, low_ff_locations = dpm.process_ISRA(df,
                                                    date1, date2,
                                                    plot=False)

    # Check that high and low values have been replaced
    assert ff_processed[3]<5
    assert ff_processed[96]>0.05

def test_extract_ISRA_modules():
    """
    Description
    -----------
        Test that we can extract module-by-module data for the ISRA
            scanner.
    """

    # Load data frame
    df = pd.read_excel(io='Fault Scanner Data 20190801 to 20210627.xlsx',
                       sheet_name='ISRA')

    # Date range we want to look at
    date1 = np.datetime64('2020-07-25')
    date2 = np.datetime64('2021-07-30')

    X, N, date_time, total_faults = dpm.extract_ISRA_modules(df, date1, date2)

    # Check size of the data
    assert np.shape(X)[0] == 5
    assert np.shape(X)[1] == N

    # Check the total_faults against manually calculated value
    assert total_faults[0] == 0.31


def test_entropy():
    """
    Description
    -----------
        Test that our method for estimating the Shannon entropy of a
            distribution is working as expected.
    """

    # Test that a histograms of all zeros reports zero entropy
    x = np.zeros(5)
    assert dpm.entropy(x) == 0

    # Test that the entropy of a uniform distribution is equal to
    # the correct value
    x = np.repeat(1, 5)
    assert np.allclose(dpm.entropy(x), np.log(5), atol=1e-10)

    # Test that the entropy of a 'concentrated' distribution is
    # equal to the correct value
    x = np.zeros(5)
    x[0] = 1
    assert np.allclose(dpm.entropy(x), np.log(1), atol=1e-10)


def test_identify_stripes():
    """
    Description
    -----------
        Applies our identify stripes method over a small period of data where
            the stripes have already been manually identified.
    """

    # Load scanner data
    scanner = 'MK4'
    ff_df = pd.read_excel('Fault Scanner Data 20190801 to 20210627.xlsx',
                          sheet_name=scanner)

    # Time period we're interested in
    initial_date = '23/08/20'
    final_date = '27/08/20'

    # True indices of stripes in this dataset
    true_indx = np.array([1, 189, 243, 245, 246])

    # Identified indices of stripes in this dataset
    stripe_locations = dpm.identify_stripes(ff_df, scanner,
                                            initial_date, final_date)
    test_indx = np.where(stripe_locations)[0]

    assert np.array_equal(test_indx, true_indx)


def test_load_raw_tag_data():
    """
    Description
    -----------
        Asserts that loading data from tag_id 135 manually and loading
            it using load_raw_tag_data gives the same results.


    """

    # True file name
    file_name = (r'Tag_Data_2020_2021/TagID 135 - FT_F160.F_CV ' +
                 'From 2020-03-22 00.00.00 to 2021-01-04 00.00.00.txt')

    # Tag ID associated with file name
    tag_id = 135

    # Load manually and using load_raw_tag_data
    df1 = pd.read_csv(file_name, sep=';')
    folder_name = 'Tag_Data_2020_2021'
    df2, load_error = dpm.load_raw_tag_data(tag_id, folder_name)

    # Check that the data in the 2 data frames are the same
    assert np.allclose(df1['AValue'].values, df2['AValue'].values)

    # Check that getting the tag id wrong returns an error
    df, load_error = dpm.load_raw_tag_data(1, folder_name)
    assert load_error is True


def test_isolate_time_period():

    """
    Description
    -----------
        Tests that the resulting date time is within the specified range,
            and that the 2 arrays are the same length.

    """

    import pandas as pd

    initial_date_time = pd.to_datetime('21/09/19', dayfirst=True)
    final_date_time = pd.to_datetime('25/10/19', dayfirst=True)

    tag_id = 135
    df, load_error = dpm.load_raw_tag_data(tag_id, 'Tag_Data_2020_2021')
    x = df['AValue'].values
    date_time = pd.to_datetime(df['DateTimeStamp'], dayfirst=True)
    x, new_date_time = dpm.isolate_time_period(x, date_time,
                                               '21/09/19',
                                               '25/10/19')

    i = new_date_time < initial_date_time
    j = new_date_time > final_date_time

    assert np.sum(i) == 0
    assert np.sum(j) == 0
    assert len(x) == len(new_date_time)


def test_eng_min_max():

    """
    Description
    -----------
        Tests that we can correctly identify signals that pass beyond
            user-defined limits.

    """

    t = np.linspace(0, 10, 50)
    x = 0.9 * np.sin(t)
    x, min_max_pass = dpm.eng_min_max(x, -1, 1)
    assert min_max_pass

    x[10] = 2
    x, min_max_pass = dpm.eng_min_max(x, -1, 1)
    assert not min_max_pass


def test_low_pass_filter():
    """
    Description
    -----------
        Test that the low-pass filter can remove high-frequency noise
            from a sinusoid.

    """

    np.random.seed(42)

    t = np.linspace(0, 20, 500)
    x = np.sin(t)
    x_noisy = x + 0.1 * np.random.randn(len(x))
    x_filt = dpm.low_pass_filter(x_noisy)

    assert np.allclose(x, x_filt, atol=0.2)


def test_high_pass_filter():
    """
    Description
    -----------
        Test that the high-pass filter can remove low-frequency signal
            from a sinusoid

    """

    t = np.linspace(0, 20, 500)
    x = np.sin(5*t)
    x_drift = x + t
    x_filt = dpm.high_pass_filter(x_drift)

    assert np.allclose(x[10::-10], x_filt[10::-10], atol=0.2)


def test_standardise():
    """
    Description
    -----------
        Test that standarisation returns signal that is zero-mean and unit
            variance.

    """

    x = 10 + 5 * np.random.randn(100)
    xs = dpm.standardise(x, np.mean(x), np.std(x))
    assert np.allclose(np.mean(xs), 0, atol=1e-10)
    assert np.allclose(np.std(xs), 1, atol=1e-10)


def test_load_time_lag():
    """
    Description
    -----------
        Test that the time lag of an example tag can be calculated
            correctly for an example tag. Is compared with a value
            that has been calculated by hand.

    """

    import pandas as pd

    # Load time lags from spreadsheet
    file_name = r'UK5 AI Furnace Model Input Pre-Processing.xlsx'
    time_lag_df = pd.read_excel(file_name, sheet_name=r'UK5 Tags')

    tag_ID = '11145'
    time_lag, load_error = dpm.load_time_lag(time_lag_df, tag_ID)

    assert time_lag == round(3 * 30)


def test_adjust_time_lag():
    """
    Description
    -----------
        Test that this method correctly adjusts for time lags and returns
            an array of the correct length.

    """

    x = np.arange(0, 10)
    x_out = dpm.adjust_time_lag(x, shift=2, to_remove=3)

    print(x_out)
    print(np.arange(1, 8))

    assert np.allclose(x_out, np.arange(1, 8))


def test_remove_spikes():
    """
    Description
    -----------
        Tests that we can remove outliers from a signal

    """

    x = 10 + np.random.randn(1000)
    x[20] = 10 + 5 * np.std(x)
    x_processed = dpm.remove_spikes(x)

    # Check that the outlier has been replaced with something else
    assert x[20] != x_processed[20]


def test_ff_remove_spikes():
    """
    Description
    -----------
        Tests that we can remove outliers from a signal sequentially

    """

    x = 5 + np.random.randn(1000)
    for i in range(20, 25):
        x[i] = 5 + 10 * np.std(x[10:19])
    x_processed = dpm.ff_remove_spikes(x)

    # Check that the outlier has been replaced with something else
    for i in range(20, 25):
        assert x[i] != x_processed[20]


def test_align_arrays():
    """
    Description
    -----------
    Just checks that the arrays coming from the method are of the
        correct size.

    """

    import pandas as pd

    # Load example data
    file_name = 'Input Post-Processing 3 ISRA.xlsx'
    X_df = pd.read_excel(file_name, sheet_name='input_data')
    Y_df = pd.read_excel(file_name, sheet_name='output_data')
    T_df = pd.read_excel(file_name, sheet_name='time_lags')

    X, Y, N, D, max_lag, time_lags = dpm.align_arrays(X_df, Y_df, T_df)

    # Check that the lengths of X and Y is as we would expect
    assert len(Y_df) == len(Y) + max_lag
    assert np.shape(X)[0] == len(Y)


def test_random_time_lags():
    """
    Description
    -----------
        Tests the method that generates random samples of time lags.

    """

    # Setting seed of random number generator so the same test is
    # performed each time.
    np.random.seed(42)

    # Load processed data
    file_name = 'Input Post-Processing 3 ISRA.xlsx'
    X_df = pd.read_excel(file_name, sheet_name='input_data')
    Y_df = pd.read_excel(file_name, sheet_name='output_data')
    T_df = pd.read_excel(file_name, sheet_name='time_lags')

    # Generate 1000 samples
    Ns = 1000
    samples = dpm.random_time_lags(T_df.values[0], Ns, s=1)

    # Check size of samples array
    assert np.shape(samples)[0] == Ns
    assert np.shape(samples)[1] == np.shape(T_df)[1]

    # Place first set of samples into the time lags data frame
    T_df.loc[0, :] = samples[0]

    # Check that we can process training data using the first sample
    X, Y, N, D, max_lag, time_lags = dpm.align_arrays(X_df, Y_df, T_df)

    # Check that the max lag is correctly identified
    assert max_lag == np.max(samples[0])
