#%%
import pandas as pd
raw_data = pd.read_csv('.\data\TrainTest_1M.csv')

#%%
enable_time_features_expansion = True
timestamp_column = 'time'
vector_keys = ['eid', 'cid', 'uid']
groupby_keys = ['src_evt', 'cat']
categorical_columns = vector_keys + groupby_keys + ['obj']
categorical_aggregators = ['unique']
numeric_columns = [timestamp_column]
numeric_aggregators = ['count', 'min', 'max', 'span']
enable_subtraction_features = True

#%%
output = {}

# Add extra time-related features based on the timestamp column
# New featuers are added in-place
def expand_time_features(data):
    # timestamp is in seconds (unix timestamp http://www.unixtimestamp.com/)
    # Features below are monotonically increasing with time, i.e. they don't
    # wrap around with respect to year, month, or week
    data[timestamp_column] = data[timestamp_column].apply(lambda t: int(t))
    data['hour'] = data[timestamp_column].apply(lambda x: int(x / 3600))
    data['day']  = data['hour'].apply(lambda x: int(x / 24))
    data['week'] = data['day'].apply(lambda x: int(x / 7))

    # Features below wrap around with respect to year, week, or day. 
    # All the features are 1-based
    data['hour_of_day'] = data['hour'].apply(lambda x: (x % 24) + 1)
    data['day_of_week'] = data['day'].apply(lambda x: (x % 7) + 1)
    data['week_of_year'] = data['week'].apply(lambda x: (x % 52) + 1)

    # Since we have time as numeric feature, it doesn't add much information to make new features numeric.
    # So we just add new features as categorical.
    categorical_columns.extend(['hour', 'day', 'week', 'hour_of_day', 'day_of_week', 'week_of_year'])
   
    # Add day_of_week as a new groupby key
    groupby_keys.append('day_of_week')

def span(arr):
    return arr.max() - arr.min()

def unique(arr):
    return pd.Series.nunique(arr)

def replace_names_in_list(input_list, replace_dict):
    for name, replacement in replace_dict.items():
        try:
            i = input_list.index(name)
            input_list[i] = replacement
        except:
          pass          

def create_aggregation_groups():
    # Replace custome aggregator names with actual functions.
    # String names are used in the the config section, b/c I wanted all the declarations to be below config.
    mapping = {'unique': unique, 'span': span}
    replace_names_in_list(categorical_aggregators, mapping)
    replace_names_in_list(numeric_aggregators, mapping)

    # Single-key aggreagtion group is for aggregation with one vector key, without any groupby key, hence single-key.
    # For single-key group, we perform both numeric and categorical aggregations.
    single_key_aggregation = [(col,categorical_aggregators) for col in categorical_columns]
    single_key_aggregation = single_key_aggregation + [(col,numeric_aggregators) for col in numeric_columns]
    single_key_aggregation_dict = dict(single_key_aggregation)

    # Double-key aggregation group is for aggregation using a vector key in combination with a groupby key, hence double-key.
    # For these aggregations, we use only numeric aggregations, to keep the number of features from exploding.
    double_key_aggregation_dict = dict((col,numeric_aggregators) for col in numeric_columns)

    aggregation_groups = [([], single_key_aggregation_dict)]
    for key in groupby_keys:
        aggregation_groups.append(([key], double_key_aggregation_dict))

    return aggregation_groups


def compute_group_features(data, group_keys, aggregation_dict):
    grouped_data = data.groupby(group_keys)
    agg_data = grouped_data.agg(aggregation_dict)
    
    if (len(group_keys) == 1):
        unstacked_data = agg_data
        formatted_columns = ['_'.join(col).strip() for col in unstacked_data.columns.values]
    else:
        unstacked_data = agg_data.unstack().reorder_levels([2,0,1], axis=1)
        column_prefix = '{0}='.format(group_keys[1])
        formatted_columns = [column_prefix + str(col[0]) + '|' + '_'.join(col[1:]).strip() for col in unstacked_data.columns.values]

    #flatten the hierarchical column index
    unstacked_data.columns = formatted_columns
    unstacked_data.sort_index(axis=1, inplace=True)
    return unstacked_data


def compute_vector_key_features(data, aggregation_groups):
    for vector_key in vector_keys:
        feature_groups = []
        for item in aggregation_groups:
            group = [vector_key] + item[0]
            aggregation_dict = item[1]
            features = compute_group_features(data, group, aggregation_dict)
            feature_groups.append(features) 

        vector_key_features = pd.concat(feature_groups, axis=1)
        output[vector_key] = vector_key_features


def add_subtraction_features(vector_keys_mapping):
    joined_output = {}
    for key, vector in output.iteritems():
        joined_output[key] = output[key].reset_index().merge(vector_keys_mapping, how='inner').set_index(vector_keys)

    # Get all the combinations of two vector_keys
    for i, key_1 in enumerate(vector_keys[:-1]):
        for key_2 in vector_keys[i+1:]:
            id = '_{0}-{1}'.format(key_1, key_2)
            output[id] = joined_output[key_1].sub(joined_output[key_2])



#%%
if enable_time_features_expansion:
    expand_time_features(raw_data)
#%%
print(raw_data.head(10))

#%%
aggregation_groups = create_aggregation_groups()
print(aggregation_groups)

#%%
vector_keys_mapping = data[vector_keys]
vector_keys_mapping = vector_keys_mapping.drop_duplicates()

compute_vector_key_features(data, aggregation_groups)

if enable_subtraction_features:
    add_subtraction_features(vector_keys_mapping)

output_vector = vector_keys_mapping
for key, vector in output.iteritems():
    # Fix column names before final merge
    key_prefix = 'key={0}|'.format(key)
    vector = vector.add_prefix(key_prefix)
    vector.reset_index(inplace=True)
    output[key] = vector

    output_vector = output_vector.merge(vector, how='left')

print(output_vector.head(50))