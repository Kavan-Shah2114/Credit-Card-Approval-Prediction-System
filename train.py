import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Starting model training and saving process...")

print("Loading and cleaning data...")
df1 = pd.read_csv('case_study1.csv')
df2 = pd.read_csv('case_study2.csv')

df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]
df = pd.merge(df1, df2, how='inner', on='PROSPECTID')

print("Selecting final features...")
columns_to_be_kept_numerical = [
    'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_open_L12M',
    'pct_tl_closed_L12M', 'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL',
    'Secured_TL', 'Unsecured_TL', 'Other_TL', 'Age_Oldest_TL', 'Age_Newest_TL',
    'time_since_recent_payment', 'max_recent_level_of_deliq', 'num_deliq_6_12mts',
    'num_times_60p_dpd', 'num_std_12mts', 'num_sub', 'num_sub_12mts', 'num_dbt',
    'num_dbt_12mts', 'num_lss', 'recent_level_of_deliq', 'enq_L3m',
    'NETMONTHLYINCOME', 'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag',
    'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever', 'HL_Flag', 'GL_Flag'
]
categorical_features = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[columns_to_be_kept_numerical + categorical_features + ['Approved_Flag']]

print("Preprocessing and encoding data...")

df.loc[df['EDUCATION'] == 'SSC', ['EDUCATION']] = 1
df.loc[df['EDUCATION'] == '12TH', ['EDUCATION']] = 2
df.loc[df['EDUCATION'] == 'GRADUATE', ['EDUCATION']] = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']] = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']] = 4
df.loc[df['EDUCATION'] == 'OTHERS', ['EDUCATION']] = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL', ['EDUCATION']] = 3
df['EDUCATION'] = df['EDUCATION'].astype(int)

df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)

joblib.dump(x.columns, 'model_columns.pkl')

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Training the final XGBoost model...")
final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    colsample_bytree=0.9,
    learning_rate=1,
    max_depth=3,
    alpha=10,
    n_estimators=100,
    random_state=42
)
final_model.fit(x, y_encoded)

print("Saving model and encoder to disk...")
joblib.dump(final_model, 'credit_card_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\nProcess complete! You now have the following files:")
print("- credit_card_model.pkl")
print("- label_encoder.pkl")
print("- model_columns.pkl")