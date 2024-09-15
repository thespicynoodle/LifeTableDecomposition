import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
import io
# Load environment variables
load_dotenv()
st.set_page_config(
    page_title="Life Expectancy and Risk Factor Decomposition Tool",
    page_icon=":world_map:",  # You can change this to any emoji or image URL
    layout="wide"  # Optional: makes the app use the full width of the browser window
)
# Supabase credentials
url = os.getenv("PROJECT_URL")
key = os.getenv("SECRET_PROJECT_API_KEY")

# Initialize Supabase client
supabase: Client = create_client(url, key)

# Define your functions (calculate_life_table and calculate_life_expectancy_contribution) here
# [Include your functions from earlier]
# Life table calculation
def calculate_life_table(deaths, population):
    """Calculate life table from deaths and population data."""
    df = pd.DataFrame({
        'Age': ['<1 year', '12-23 months', '2-4 years', '5-9 years', '10-14 years', '15-19 years', '20-24 years', 
                '25-29 years', '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years', 
                '55-59 years', '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', 
                '85-89 years', '90-94 years', '95+ years'],
        'Years in Interval (n)': [1, 1, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
        'Deaths (nDx)': deaths,
        'Reported Population (nNx)': population
    })

    df['Mortality Rate (nmx)'] = df['Deaths (nDx)'] / df['Reported Population (nNx)']
    
    # Add linearly adjusted probabilities
    df['Linearity Adjustment (nax)'] = 0.5
    df.loc[0, 'Linearity Adjustment (nax)'] = 0.1
    df.loc[1, 'Linearity Adjustment (nax)'] = 0.3
    df.loc[2, 'Linearity Adjustment (nax)'] = 0.4

    df['Probability of Dying (nqx)'] = df['Years in Interval (n)'] * df['Mortality Rate (nmx)'] / \
                                       (1 + (1 - df['Linearity Adjustment (nax)']) * df['Mortality Rate (nmx)'] * df['Years in Interval (n)'])
    
    df['Probability of Surviving (npx)'] = 1 - df['Probability of Dying (nqx)']
    
    df['Individuals Surviving (lx)'] = 100000
    for i in range(1, len(df)):
        df.loc[i, 'Individuals Surviving (lx)'] = df.loc[i - 1, 'Individuals Surviving (lx)'] * df.loc[i - 1, 'Probability of Surviving (npx)']
    
    df['Deaths in Interval (ndx)'] = df['Individuals Surviving (lx)'] * df['Probability of Dying (nqx)']
    df.at[len(df)-1, 'Deaths in Interval (ndx)'] = df.at[len(df)-1, 'Individuals Surviving (lx)']

    df['Years Lived in Interval (nLx)'] = df['Years in Interval (n)'] * ((df['Individuals Surviving (lx)'] + df['Individuals Surviving (lx)'].shift(-1)) / 2)
    df.at[0, 'Years Lived in Interval (nLx)'] = df.at[0,'Years in Interval (n)'] * (df.at[1, 'Individuals Surviving (lx)'] + (df.at[0, 'Linearity Adjustment (nax)'] * df.at[0, 'Deaths in Interval (ndx)'])) 
    df.at[1, 'Years Lived in Interval (nLx)'] = df.at[1, 'Years in Interval (n)'] * (df.at[2, 'Individuals Surviving (lx)'] + (df.at[1, 'Linearity Adjustment (nax)'] * df.at[1, 'Deaths in Interval (ndx)'])) 
    df.at[2, 'Years Lived in Interval (nLx)'] = df.at[2, 'Years in Interval (n)'] * (df.at[3, 'Individuals Surviving (lx)'] + (df.at[2, 'Linearity Adjustment (nax)'] * df.at[2, 'Deaths in Interval (ndx)']))
    df.at[len(df)-1, 'Years Lived in Interval (nLx)'] = df.at[len(df)-1, 'Individuals Surviving (lx)'] / df.at[len(df)-1, 'Mortality Rate (nmx)']
    
    df['Cumulative Years Lived (Tx)'] = df['Years Lived in Interval (nLx)'][::-1].cumsum()[::-1]
    
    df['Expectancy of Life at Age x (ex)'] = df['Cumulative Years Lived (Tx)'] / df['Individuals Surviving (lx)']
    
    return df

# Decomposition calculation
def calculate_life_expectancy_contribution(life_table_1, life_table_2):
    """Calculate the contribution of each age group to life expectancy difference between two years."""
    # Ensure that the dataframes are aligned on age groups
    if not (life_table_1['Age'].equals(life_table_2['Age'])):
        raise ValueError("Age groups in the two life tables must match")

    contributions = []

    for i in range(len(life_table_1)):
        if i == len(life_table_1) - 1:  # Last age group (open-ended)
            delta_x = (life_table_1.loc[i, 'Individuals Surviving (lx)'] / life_table_1.loc[0, 'Individuals Surviving (lx)']) * \
                      (life_table_2.loc[i, 'Cumulative Years Lived (Tx)'] / life_table_2.loc[i, 'Individuals Surviving (lx)'] - \
                       life_table_1.loc[i, 'Cumulative Years Lived (Tx)'] / life_table_1.loc[i, 'Individuals Surviving (lx)'])
        else:
            # First term
            first_term = (life_table_1.loc[i, 'Individuals Surviving (lx)'] / life_table_1.loc[0, 'Individuals Surviving (lx)']) * \
                         (life_table_2.loc[i, 'Years Lived in Interval (nLx)'] / life_table_2.loc[i, 'Individuals Surviving (lx)'] - \
                          life_table_1.loc[i, 'Years Lived in Interval (nLx)'] / life_table_1.loc[i, 'Individuals Surviving (lx)'])
            
            # Second term
            second_term = (life_table_2.loc[i+1, 'Cumulative Years Lived (Tx)'] / life_table_1.loc[0, 'Individuals Surviving (lx)']) * \
                          ((life_table_1.loc[i, 'Individuals Surviving (lx)'] / life_table_2.loc[i, 'Individuals Surviving (lx)']) - \
                           (life_table_1.loc[i+1, 'Individuals Surviving (lx)'] / life_table_2.loc[i+1, 'Individuals Surviving (lx)']))
            
            delta_x = first_term + second_term

        contributions.append(delta_x)

    # Create a DataFrame for the contributions
    contribution_df = pd.DataFrame({
        'Age': life_table_1['Age'],
        'Contribution to LE difference (years)': contributions
    })

    # Add a row for the sum of contributions
    total_contribution = sum(contributions)
    total_row = pd.DataFrame({
        'Age': ['Life expectancy difference'],
        'Contribution to LE difference (years)': [total_contribution]
    })
    contribution_df = pd.concat([contribution_df, total_row], ignore_index=True)

    return contribution_df
#@st.cache_data 
# Define the list of countries
countries = [
    'Australia',
    'Japan',
    'Canada',
    'United States of America',
    'Federal Republic of Germany',
    'Kingdom of the Netherlands',
    'Kingdom of Denmark',
    'French Republic',
    'Kingdom of Belgium',
    'Kingdom of Norway',
    'Kingdom of Sweden',
    'Republic of Austria',
    'Republic of Finland',
    'Republic of Korea',
    'United Kingdom of Great Britain and Northern Ireland',
    'Swiss Confederation'
]

# Define the list of years from 1990 to 2021
years = list(range(1990, 2022))

# Define genders (adjust according to your data)
genders = ['Male', 'Female']

st.sidebar.title('Filters')

# Country selector
country = st.sidebar.selectbox('Select Country', countries)

# Gender selector
gender = st.sidebar.selectbox('Select Gender', genders)

# Year selectors
year1 = st.sidebar.selectbox('Select Year 1', years, index=0)
year2 = st.sidebar.selectbox('Select Year 2', years, index=1)

def get_data(country, gender, year):
    response = supabase.table('PopulationData').select('*') \
        .eq('location_name', country) \
        .eq('sex_name', gender) \
        .eq('year', year) \
        .execute()
    data = pd.DataFrame(response.data)
    return data

data_year1 = get_data(country, gender, year1)
data_year2 = get_data(country, gender, year2)

age_groups = ['<1 year', '12-23 months', '2-4 years', '5-9 years', '10-14 years', '15-19 years', '20-24 years', 
              '25-29 years', '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years', 
              '55-59 years', '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', 
              '85-89 years', '90-94 years', '95+ years']

def get_deaths_and_population(data):
    data = data.set_index('age_name').reindex(age_groups).reset_index()
    data = data.fillna(0)
    deaths = data['total_deaths'].tolist()
    population = data['population'].tolist()
    # Include risk factor deaths
    tobacco_deaths = data['tobacco_deaths'].tolist()
    alcohol_deaths = data['alc_deaths'].tolist()
    drug_deaths = data['drug_deaths'].tolist()
    return deaths, population, tobacco_deaths, alcohol_deaths, drug_deaths

deaths1, population1, tobacco_deaths1, alcohol_deaths1, drug_deaths1 = get_deaths_and_population(data_year1)
deaths2, population2, tobacco_deaths2, alcohol_deaths2, drug_deaths2 = get_deaths_and_population(data_year2)

life_table1 = calculate_life_table(deaths1, population1)
life_table2 = calculate_life_table(deaths2, population2)

contribution_df = calculate_life_expectancy_contribution(life_table1, life_table2)

# Update the 'Age' column to categorical and sort
contribution_df['Age'] = pd.Categorical(contribution_df['Age'], categories=age_groups + ['Life expectancy difference'], ordered=True)
contribution_df = contribution_df.sort_values('Age')

def calculate_risk_factor_proportions_by_age(total_deaths, tobacco_deaths, alcohol_deaths, drug_deaths):
    other_deaths = [td - (tob + alc + drug) for td, tob, alc, drug in zip(total_deaths, tobacco_deaths, alcohol_deaths, drug_deaths)]
    
    proportions = []
    for age, td, tob, alc, drug, oth in zip(age_groups, total_deaths, tobacco_deaths, alcohol_deaths, drug_deaths, other_deaths):
        if td > 0:
            tob_prop = (tob / td) 
            alc_prop = (alc / td) 
            drug_prop = (drug / td) 
            oth_prop = (oth / td) 
        else:
            tob_prop = alc_prop = drug_prop = oth_prop = 0
        
        proportions.append({
            'Age': age,
            'Tobacco': tob_prop,
            'Alcohol': alc_prop,
            'Drugs': drug_prop,
            'Other Risk Factors': oth_prop
        })
    
    df = pd.DataFrame(proportions)
    return df

risk_factors_by_age1 = calculate_risk_factor_proportions_by_age(deaths1, tobacco_deaths1, alcohol_deaths1, drug_deaths1)
risk_factors_by_age2 = calculate_risk_factor_proportions_by_age(deaths2, tobacco_deaths2, alcohol_deaths2, drug_deaths2)

def plot_risk_factors_by_age(df, year):
    df['Age'] = pd.Categorical(df['Age'], categories=age_groups, ordered=True)
    df = df.sort_values('Age')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Age'],
        y=df['Tobacco'],
        name='Tobacco',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=df['Age'],
        y=df['Alcohol'],
        name='Alcohol',
        marker_color='lightsalmon'
    ))
    fig.add_trace(go.Bar(
        x=df['Age'],
        y=df['Drugs'],
        name='Drugs',
        marker_color='gold'
    ))
    fig.add_trace(go.Bar(
        x=df['Age'],
        y=df['Other Risk Factors'],
        name='Other Risk Factors',
        marker_color='lightgreen'
    ))

    fig.update_layout(
        barmode='stack',
        xaxis_title='Age Group',
        yaxis_title='Proportion of Deaths',
        title=f'Risk Factor Proportions by Age Group for Year {year}',
        xaxis={'categoryorder':'array', 'categoryarray': age_groups}
    )
    return fig

# Calculate Risk Factor Contributions to Life Expectancy Change

# 1. Extract mortality rates
mortality_rates1 = life_table1[['Age', 'Mortality Rate (nmx)']].copy()
mortality_rates2 = life_table2[['Age', 'Mortality Rate (nmx)']].copy()
mortality_rates1.rename(columns={'Mortality Rate (nmx)': 'Mortality Rate Year 1'}, inplace=True)
mortality_rates2.rename(columns={'Mortality Rate (nmx)': 'Mortality Rate Year 2'}, inplace=True)

# 2. Get risk factor proportions
#risk_factors_by_age1.columns = ['Age', 'Tobacco Proportion Year 1', 'Alcohol Proportion Year 1', 'Drugs Proportion Year 1', 'Other Risk Factors Proportion Year 1']
#risk_factors_by_age2.columns = ['Age', 'Tobacco Proportion Year 2', 'Alcohol Proportion Year 2', 'Drugs Proportion Year 2', 'Other Risk Factors Proportion Year 2']
# Create copies and rename columns for merging
risk_factors_by_age1_renamed = risk_factors_by_age1.copy()
risk_factors_by_age1_renamed.columns = ['Age', 'Tobacco Proportion Year 1', 'Alcohol Proportion Year 1', 'Drugs Proportion Year 1', 'Other Risk Factors Proportion Year 1']

risk_factors_by_age2_renamed = risk_factors_by_age2.copy()
risk_factors_by_age2_renamed.columns = ['Age', 'Tobacco Proportion Year 2', 'Alcohol Proportion Year 2', 'Drugs Proportion Year 2', 'Other Risk Factors Proportion Year 2']
# 3. Extract delta_x
delta_x_df = contribution_df[contribution_df['Age'] != 'Life expectancy difference'][['Age', 'Contribution to LE difference (years)']].copy()
delta_x_df.rename(columns={'Contribution to LE difference (years)': 'Delta e_x'}, inplace=True)

# 4. Merge all data
merged_df = mortality_rates1.merge(mortality_rates2, on='Age')
merged_df = merged_df.merge(risk_factors_by_age1_renamed, on='Age')
merged_df = merged_df.merge(risk_factors_by_age2_renamed, on='Age')
merged_df = merged_df.merge(delta_x_df, on='Age')

# Convert proportions to decimals
#for col in ['Tobacco Proportion Year 1', 'Alcohol Proportion Year 1', 'Drugs Proportion Year 1', 'Other Risk Factors Proportion Year 1',
 #           'Tobacco Proportion Year 2', 'Alcohol Proportion Year 2', 'Drugs Proportion Year 2', 'Other Risk Factors Proportion Year 2']:
 #   merged_df[col] = merged_df[col] 

# Compute risk factor contributions
risk_factors = ['Tobacco', 'Alcohol', 'Drugs', 'Other Risk Factors']
contributions = []

for index, row in merged_df.iterrows():
    mortality_rate_year1 = row['Mortality Rate Year 1']
    mortality_rate_year2 = row['Mortality Rate Year 2']
    delta_e_x = row['Delta e_x']
    age = row['Age']
    
    mortality_rate_diff = mortality_rate_year2 - mortality_rate_year1
    
    if mortality_rate_diff == 0:
        # Avoid division by zero
        continue
    
    for risk_factor in risk_factors:
        proportion_year1 = row[f'{risk_factor} Proportion Year 1']
        proportion_year2 = row[f'{risk_factor} Proportion Year 2']
        
        numerator = (mortality_rate_year2 * proportion_year2) - (mortality_rate_year1 * proportion_year1)
        denominator = mortality_rate_diff
        
        contribution = (numerator / denominator) * delta_e_x if denominator != 0 else 0
        
        contributions.append({
            'Age': age,
            'Risk Factor': risk_factor,
            'Contribution to LE difference (years)': contribution
        })

risk_factor_contributions_df = pd.DataFrame(contributions)

# Pivot the DataFrame
pivot_df = risk_factor_contributions_df.pivot(index='Age', columns='Risk Factor', values='Contribution to LE difference (years)').reset_index()

# Fill NaN values with zero
pivot_df.fillna(0, inplace=True)

# Calculate total contributions
total_contributions = pivot_df[risk_factors].sum().to_frame().T
total_contributions['Age'] = 'Total'

# Append total contributions
pivot_df = pd.concat([pivot_df, total_contributions], ignore_index=True)

# Reorder columns
pivot_df = pivot_df[['Age'] + risk_factors]

# Set 'Age' as a categorical variable with the specified order
pivot_df['Age'] = pd.Categorical(pivot_df['Age'], categories=age_groups + ['Total'], ordered=True)

# Sort the DataFrame based on 'Age'
pivot_df = pivot_df.sort_values('Age')






# Streamlit App Layout

st.title('Life Expectancy Analysis Dashboard')

st.header(f'Analysis for {country} - {gender}')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Life Tables', 'Life Expectancy Change Contribution', 'Risk Factor Proportions', 'Risk Factor Contributions to LE Change', 'Raw Data'])

with tab1:
    st.subheader(f'Life Table for Year {year1}')
    st.dataframe(life_table1)

    st.subheader(f'Life Table for Year {year2}')
    st.dataframe(life_table2)

with tab2:
    st.subheader('Contribution of Age Groups to Life Expectancy Change')

    # Exclude the total row from the age group plotting
    contribution_plot_df = contribution_df[contribution_df['Age'] != 'Life expectancy difference']

    # Create the bar chart using Plotly Express
    fig = px.bar(
        contribution_plot_df,
        x='Age',
        y='Contribution to LE difference (years)',
        title='Contribution of Age Groups to Life Expectancy Change',
        labels={'Contribution to LE difference (years)': 'Years'}
    )
    st.plotly_chart(fig)

    # Display the total life expectancy difference
    total_difference = contribution_df[contribution_df['Age'] == 'Life expectancy difference']['Contribution to LE difference (years)'].values[0]
    st.write(f"**Total Life Expectancy Difference:** {total_difference:.4f} years")

    # **Add the DataFrame below the chart**
    st.write("### Life Expectancy Contribution Data")
    # Round the contributions to 4 decimal places for readability
    contribution_df_display = contribution_df.copy()
    contribution_df_display['Contribution to LE difference (years)'] = contribution_df_display['Contribution to LE difference (years)'].round(4)
    st.dataframe(contribution_df_display)

with tab3:
    st.subheader(f'Risk Factor Proportions by Age Group for Year {year1}')
    st.dataframe(risk_factors_by_age1)
    st.plotly_chart(plot_risk_factors_by_age(risk_factors_by_age1, year1))

    st.subheader(f'Risk Factor Proportions by Age Group for Year {year2}')
    st.dataframe(risk_factors_by_age2)
    st.plotly_chart(plot_risk_factors_by_age(risk_factors_by_age2, year2))

with tab4:
    st.subheader('Risk Factor Contributions to Life Expectancy Change')
    st.dataframe(pivot_df)

    # Exclude the total row for plotting
    plot_df = pivot_df[pivot_df['Age'] != 'Total']

    # Ensure 'Age' is a categorical variable ordered from youngest to oldest
    plot_df['Age'] = pd.Categorical(plot_df['Age'], categories=age_groups, ordered=True)

    # Sort the DataFrame by 'Age'
    plot_df = plot_df.sort_values('Age')

    # Melt the DataFrame for plotting
    plot_df_melted = plot_df.melt(id_vars='Age', value_vars=risk_factors, var_name='Risk Factor', value_name='Contribution')

    # Sort the melted DataFrame by 'Age'
    plot_df_melted = plot_df_melted.sort_values('Age')

    # Create a stacked bar chart
    fig = px.bar(
        plot_df_melted,
        x='Age',
        y='Contribution',
        color='Risk Factor',
        title='Risk Factor Contributions to Life Expectancy Change by Age Group',
        labels={'Contribution': 'Contribution to LE Difference (years)'},
        category_orders={'Age': age_groups}
    )
    st.plotly_chart(fig)

with tab5:
    st.subheader('Raw Data from Supabase')

    st.write(f"**Data for Year {year1}:**")
    st.dataframe(data_year1)

    st.write(f"**Data for Year {year2}:**")
    st.dataframe(data_year2)

# Create a BytesIO buffer to hold the Excel file in memory
output = io.BytesIO()

# Use pandas ExcelWriter to write dataframes to different sheets
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    # Write each DataFrame to a specific sheet
    life_table1.to_excel(writer, sheet_name=f'Life Table {year1}', index=False)
    life_table2.to_excel(writer, sheet_name=f'Life Table {year2}', index=False)
    contribution_df.to_excel(writer, sheet_name='LE Contribution', index=False)
    risk_factors_by_age1.to_excel(writer, sheet_name=f'Risk Factors {year1}', index=False)
    risk_factors_by_age2.to_excel(writer, sheet_name=f'Risk Factors {year2}', index=False)
    pivot_df.to_excel(writer, sheet_name='Risk Factor LE Change', index=False)
    data_year1.to_excel(writer, sheet_name=f'Raw Data {year1}', index=False)
    data_year2.to_excel(writer, sheet_name=f'Raw Data {year2}', index=False)
    # No need to call writer.save() here

# Set the position to the beginning of the stream
output.seek(0)

# Use st.download_button to allow the user to download the Excel file
st.download_button(
    label="Download Results as Excel",
    data=output,
    file_name='analysis_results.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
