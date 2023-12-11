#!/usr/bin/env python
# coding: utf-8

# # Predicción de Desarrollo Humano con base en la Inserción de la mujer

# Empezamos por la búsqueda de Data Sets adecuados para la predicción. Finalmente he encontrado un DataSet de World Bank / Gender Bank que combina datos de genero con el desarrollo económico, con todos los países y desde 19901 a 2021. A partir de este DataSet voy a crear la predicción de Desarrollo.
# 
# El Data Set está compuesto de columnas con distintos marcadores, que se repiten con años diferentes. Ejemplo: "Years of schooling"(YOS) se repite para todos los años, esto nos da las columnas YOS_1991, YOS_1992,....,YOS_2021. 

# Importamos las bibliotecas y el Data Set.
# Creamos un filtro para eliminar columnas que no nos interesan.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
data = pd.read_csv("HDR21-22_Composite_indices_complete_time_series.csv", header = 0)
df = pd.DataFrame(data)

df.drop(df.filter(regex='region|gdi_group_|lfpr_f_|gni_pc_f_|ineq_edu_|hdi_f|ineq_le_|coef_ineq_|hdi_rank_|ihdi|ifdi_f|gii_|gdi_|iso3|mys_f|ineq_inc|hdicode|loss|gii_rank|hdi_m|le_m|eys_m|mys_m|se_m|lfpr_m|pr_m|gni_pc_m|rankdiff_hdi_phdi|diff_hdi_phdi|mf|phdi').columns, axis=1, inplace=True) 
#Con Regex, he creado un filtro para eliminar columnas que no me interesan y hacer el df más pequeño
#print(df)


# Cambiar los años a una columna y unificar las columnas qu están repartidas por años en una variable. Seguimos con el EDA

# In[2]:


# Utilizamos melt para transformar los datos, unificando las columnas con el año. Creamos una columna año

df_melted = pd.melt(
    df,
    id_vars=['country','continent'],
    var_name='variable',
    value_name='valor'
)
# Extraemos el año de la columna 'variable' y creamos una nueva columna 'año'
df_melted['año'] = df_melted['variable'].str.extract(r'(\d{4})')
# Extraemos el prefijo de la columna 'variable' y creamos una nueva columna 'indicador'
df_melted['indicador'] = df_melted['variable'].str.replace(r'\d+','') #queda sin el _
# Eliminamos la columna 'variable'
df_melted = df_melted.drop('variable', axis=1)
# Creamos columnas para cada indicador usando pivot_table
df_final = df_melted.pivot_table(index=[ 'country', 'continent', 'año'],
                                columns='indicador',
                                values='valor',
                                aggfunc='first').reset_index()
# Reorganizamos las columnas
#df_final = df_final[['iso3', 'country', 'hdicode', 'region', 'hdi_rank_2021', 'año', 'hdi','le', 'eys', 'gnipc', 'gdi','ihdi','gii', 'mmr','abr','hdif']]
# Mostramos el resultado
df_final


# 'gdi_group', 'hdi_f', 'le_f', 'eys_f', 'mys_f', 'ineq_edu','hdi_f' ,'coef_ineq', 'gni_pc_f','coef_ineq', 'ineq_le', 'se_f', 'pr_f', 'lfpr_f','co2_prod'




# In[3]:


#cuantas lineas y columnas tiene mi df
df_final.shape


# In[4]:


#cambio visualizacion para ver todas lineas y columnas
pd.options.display.max_columns = None

pd.options.display.max_rows = None


# In[ ]:


#df_new.isnull().sum()


# In[5]:


#transformo la columna año en INT
df_final.año = df_final.año.astype(int)


# Hay muchos valores NaN en los primeros años y hay países de los cuales no se pueden encontrar datos de los primeros años. Decido eliminar los años previos a 2000 para trabajar con un df más consistente. 

# In[6]:


df_new = df_final[df_final['año']>= 2000]


# In[6]:


#ver numero columnas y filas
df_new.shape


# Decido eliminar paises que no tienens datos de calidad (NaN) o tienen datos muy poco fiables para el EDA.
# 
# 
# 

# Sigo con la limpieza y preparación de los datos. Hay paises que tienen muchos campos NaN , y no disponen de datos en la red (siempre busco en páginas fiables, como UNICEF, ONU, GenderData). Decido eliminar algunos paises que no tienen datos de calidad o fiables (ej: North Korea)

# In[7]:


#quito 'San Marino' de la lista
df_new= df_new[df_new.country!='San Marino']


# In[8]:


#quito 'Monaco' de la lista
df_new= df_new[df_new.country!='Monaco']


# In[9]:


df_new= df_new[df_new.country!="Korea (Democratic People's Rep. of)"]


# In[10]:


df_new= df_new[df_new.country!='Hong Kong, China (SAR)']


# In[11]:


df_new= df_new[df_new.country!='Liechtenstein']


# In[ ]:


#df_new.isnull().sum()


# Sigo analizando los NaN por columna y pais (ej:"co_prod" de Timor Leste)  y quiero ver si los puedo igualar a los valores siguientes. Lo aplico a distintos paises.
# 
# 
# 

# In[12]:


#miro co_prod de Timo Leste, ya que tiene Nan en unos años y quiero ver si los puedo igualar a los valores siguientes. 
df_new[(df_new.country == 'Timor-Leste') & (df_new.año <= 2005)]


# In[13]:


#le digo que coja el primer valor TRUE para fillna los Nan anteriores
df_new.loc[df_new.country == 'Timor-Leste'] = df_new[df_new.country == 'Timor-Leste'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[16]:


#df_new[(df_new.country == 'Afghanistan') & (df_new.año <= 2005)]


# In[17]:


#df_new[(df_new.country == 'Afganistan') & (df_new.año <= 2009)]
df_new.loc[df_new.country == 'Afghanistan'] = df_new[df_new.country == 'Afghanistan'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[20]:


df_new.loc[df_new.country == 'Antigua and Barbuda'] = df_new[df_new.country == 'Antigua and Barbuda'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[21]:


#df_new[(df_new.country == 'South Sudan') & (df_new.año <= 2015)]


# In[22]:


df_new.loc[df_new.country == 'South Sudan'] = df_new[df_new.country == 'South Sudan'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[23]:


#df_new[df_new['mmr_'].isna()]


# In[24]:


#df_new[df_new.mmr_.isnull()].country.unique()


# In[25]:


df_new.loc[df_new.country == 'Marshall Islands'] = df_new[df_new.country == 'Marshall Islands'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[26]:


df_new.loc[df_new.country == 'Afganistan'] = df_new[df_new.country == 'Afganistan'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[27]:


df_new.loc[df_new.country == 'Chad'] = df_new[df_new.country == 'Chad'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[28]:


df_new.loc[df_new.country == 'Bhutan'] = df_new[df_new.country == 'Bhutan'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[29]:


df_new.loc[df_new.country == 'Brazil'] = df_new[df_new.country == 'Brazil'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[30]:


df_new.loc[df_new.country == 'Ecuador'] = df_new[df_new.country == 'Ecuador'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[31]:


df_new.loc[df_new.country == 'Eritrea'] = df_new[df_new.country == 'Eritrea'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[32]:


df_new.loc[df_new.country == 'Ethiopia'] = df_new[df_new.country == 'Ethiopia'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[33]:


df_new.loc[df_new.country == 'Grenada'] = df_new[df_new.country == 'Grenada'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[34]:


df_new.loc[df_new.country == 'South Sudan'] = df_new[df_new.country == 'South Sudan'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[35]:


df_new.loc[df_new.country == 'Haiti'] = df_new[df_new.country == 'Haiti'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[36]:


df_new.loc[df_new.country == 'Montenegro'] = df_new[df_new.country == 'Montenegro'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[37]:


df_new.loc[df_new.country == 'Nauru'] = df_new[df_new.country == 'Nauru'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[38]:


df_new.loc[df_new.country == 'Myanmar'] = df_new[df_new.country == 'Myanmar'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[39]:


df_new.loc[df_new.country == 'Congo (Democratic Republic of the)'] = df_new[df_new.country == 'Congo (Democratic Republic of the)'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[40]:


df_new.loc[df_new.country == 'Bosnia and Herzegovina'] = df_new[df_new.country == 'Bosnia and Herzegovina'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[41]:


df_new.loc[df_new.country == 'Nicaragua'] = df_new[df_new.country == 'Nicaragua'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[42]:


df_new.loc[df_new.country == 'Nigeria'] = df_new[df_new.country == 'Nigeria'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[43]:


df_new.loc[df_new.country == 'Oman'] = df_new[df_new.country == 'Oman'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[44]:


df_new.loc[df_new.country == 'Palau'] = df_new[df_new.country == 'Palau'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[45]:


df_new.loc[df_new.country == 'Palestine'] = df_new[df_new.country == 'Palestine'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[46]:


df_new.loc[df_new.country == 'Serbia'] = df_new[df_new.country == 'Serbia'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[47]:


df_new.loc[df_new.country == 'Samoa'] = df_new[df_new.country == 'Samoa'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[48]:


df_new.loc[df_new.country == 'Paraguay'] = df_new[df_new.country == 'Paraguay'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[49]:


df_new.loc[df_new.country == 'Peru'] = df_new[df_new.country == 'Peru'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[50]:


df_new.loc[df_new.country == 'Guinea-Bissau'] = df_new[df_new.country == 'Guinea-Bissau'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[51]:


df_new.loc[df_new.country == 'Russian Federation'] = df_new[df_new.country == 'Russian Federation'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[52]:


df_new.loc[df_new.country == 'Saint Vincent and the Grenadines'] = df_new[df_new.country == 'Saint Vincent and the Grenadines'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[53]:


df_new.loc[df_new.country == 'Saint Kitts and Nevis'] = df_new[df_new.country == 'Saint Kitts and Nevis'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[54]:


df_new.loc[df_new.country == 'Libya'] = df_new[df_new.country == 'Libya'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[55]:


df_new.loc[df_new.country == 'Andorra'] = df_new[df_new.country == 'Andorra'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[56]:


df_new.loc[df_new.country == 'Marshall Islands'] = df_new[df_new.country == 'Marshall Islands'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[57]:


df_new.loc[df_new.country == 'Bahrain'] = df_new[df_new.country == 'Bahrain'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[58]:


df_new.loc[df_new.country == 'Belarus'] = df_new[df_new.country == 'Belarus'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[59]:


df_new.loc[df_new.country == 'Brunei Darussalam'] = df_new[df_new.country == 'Brunei Darussalam'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[60]:


df_new.loc[df_new.country == 'Somalia'] = df_new[df_new.country == 'Somalia'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[61]:


df_new.loc[df_new.country == 'Suriname'] = df_new[df_new.country == 'Suriname'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[62]:


df_new.loc[df_new.country == 'Turkmenistan'] = df_new[df_new.country == 'Turkmenistan'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[63]:


df_new.loc[df_new.country == 'Vanuatu'] = df_new[df_new.country == 'Vanuatu'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[64]:


df_new.loc[df_new.country == 'Lebanon'] = df_new[df_new.country == 'Lebanon'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[65]:


df_new.loc[df_new.country == 'Palestine, State of'] = df_new[df_new.country == 'Palestine, State of'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[66]:


#df_new[df_new.pr_f_.isnull()].country.unique()


# In[67]:


#df_new[df_new.gnipc_.isnull()].country.unique()


# In[68]:


#df_new


# In[69]:


#df_new.loc[df['country'] == 'Andorra', 'eys_f_'] = df_new.loc[df['country'] == 'Andorra', 'eys_f_'].fillna(13.3)
mask = (df_new['country'] == 'Andorra')
df_new.loc[mask, 'eys_f_'] = df_new.loc[mask, 'eys_f_'].fillna(13.3)
df_new.loc[mask, 'mmr_'] = df_new.loc[mask, 'mmr_'].fillna(0)


# In[70]:


mask = (df_new['country'] == 'Dominica')
df_new.loc[mask, 'eys_f_'] = df_new.loc[mask, 'mmr_'].fillna(11.3)


# In[71]:


mask = (df_new['country'] == 'Bahamas')
df_new.loc[mask, 'eys_f_'] = df_new.loc[mask, 'eys_f_'].fillna(11.3)


# In[72]:


mask = (df_new['country'] == 'Equatorial Guinea')
df_new.loc[mask, 'eys_f_'] = df_new.loc[mask, 'eys_f_'].fillna(5)


# In[73]:


mask = (df_new['country'] == 'Somalia')
df_new.loc[mask, 'eys_f_'] = df_new.loc[mask, 'eys_f_'].fillna(2)


# In[74]:


mask = (df_new['country'] == 'Somalia')
df_new.loc[mask, 'eys_'] = df_new.loc[mask, 'eys_'].fillna(5.1)


# In[75]:


mask = (df_new['country'] == 'Somalia')
df_new.loc[mask, 'mys_'] = df_new.loc[mask, 'mys_'].fillna(3.5)


# In[76]:


mask = (df_new['country'] == 'Qatar')
df_new.loc[mask, 'pr_f_'] = df_new.loc[mask, 'pr_f_'].fillna(60.5)


# In[77]:


mask = (df_new['country'] == 'Saint Kitts and Nevis')
df_new.loc[mask, 'mmr_'] = df_new.loc[mask, 'mmr_'].fillna(2)


# In[78]:


df_new[df_new.mmr_.isnull()].country.unique()


# In[79]:


mask = (df_new['country'] == 'Dominica')
df_new.loc[mask, 'mmr_'] = df_new.loc[mask, 'mmr_'].fillna(33.4)

#mask = (df_new['country'] == 'Dominica') & (df_new['año'] < 2005)
#df_new.loc[mask] = df_new.loc[mask].sort_values(by='año', ascending=False).fillna(method='ffill')


# In[80]:


#df_new[(df_new.country == 'Guinea-Bissau') & (df_new.año <= 2008)]


# In[81]:


df_new.loc[df_new.country == 'Guinea-Bissau'] = df_new[df_new.country == 'Guinea-Bissau'].sort_values(by='año',ascending= False).fillna(method='ffill') 


# In[82]:


mask = (df_new['country'] == 'Micronesia (Federated States of)')
df_new.loc[mask, 'eys_f_'] = df_new.loc[mask, 'pr_f_'].fillna(5.692)


# In[83]:


mask = (df_new['country'] == 'Palestine, State of')
df_new.loc[mask, 'pr_f_'] = df_new.loc[mask, 'pr_f_'].fillna(13.77)


# In[84]:


#df_new[df_new.hdi_.isnull()].country.unique()


# In[85]:


mask = (df_new['country'] == 'Saudi Arabia')
df_new.loc[mask, 'pr_f_'] = df_new.loc[mask, 'pr_f_'].fillna(23)


# In[86]:


mask = (df_new['country'] == 'Somalia')
df_new.loc[mask, 'hdi_'] = df_new.loc[mask, 'hdi_'].fillna(0.364)


# In[87]:


mask = (df_new['country'] == 'Nauru')
df_new.loc[mask, 'hdi_'] = df_new.loc[mask, 'hdi_'].fillna(0.72)


# In[88]:


mask = (df_new['country'] == 'Nauru')
df_new.loc[mask, 'mys_'] = df_new.loc[mask, 'mys_'].fillna(6.5)


# In[89]:


#df_new.isnull().sum()


# In[90]:


mask = (df_new['country'] == 'Libya')
df_new.loc[mask, 'pr_f_'] = df_new.loc[mask, 'pr_f_'].fillna(4.736842)


# In[91]:


#df_new


# Cambio los nombre de las columnas para que se entienda mejor.

# In[92]:


df_new.columns


# Gross National Income (GNI) per capita. GNI per capita is a measure of the average income of a country's residents. It is calculated by dividing a country's GNI by its population. This metric provides insight into the overall economic well-being of a country's citizens.
# 
# Mean Years of Schooling. This variable typically represents the average number of years of education received by a country's population. It is a measure of the educational attainment or literacy level of the population.
# 
# Expected Years of Schooling. Expected Years of Schooling is a measure of the number of years of education a child entering school can expect to receive, assuming that current enrollment and educational attainment patterns remain constant. It provides a projection of the educational opportunities available to the younger generation.
# 
# Life Expectancy at Birth. Life expectancy represents the average number of years a person can expect to live from birth. It is a key indicator of the overall health and quality of life in a country.
# 
# Human Development Index. The HDI is a composite index that measures overall human development by combining indicators of income, education, and health. It provides a broad assessment of a country's development status.
# 
# Expected Years of Schooling for Females. Similar to "eys," it represents the number of years of education that a female child entering school can expect to receive.
# 
# Mean Years of Schooling for Females. This variable probably represents the average number of years of education received by the female population.
# 
# Maternal Mortality Rate. The number of maternal deaths per 100,000 live births.
# 
# Adolescent Birth Rate. The rate of births among adolescent females.
# 
# Share of Seats in Parliament held by Females.
# 
# Participation Rate of Females in the Labor Force.
# 
# "co2_prod": CO2 Production or Carbon Dioxide Emissions.
# 
# 

# In[93]:


# Cambiar el nombre de las columnas
nuevos_nombres = ['Country', 'Continent', 'Year', 'Adolescent Birth Rate', 'co_prod_', 'Expected Years of Schooling', 'Expected Years of Schooling for Females',
       'Gross National Income per capita', 'Human Development Index', 'Life Expectancy at Birth', 'Females Life Expectancy at Birth', 'Maternal Mortality Rate', 'Mean Years of Schooling', 'Participation Rate of Females in the Labor Force', 'Female Share of Seats in Parliament']
df_new.columns = nuevos_nombres

# Mostrar el DataFrame con los nombres de columnas cambiados
print("\nDataFrame con nombres de columnas cambiados:")
print(df_new)


# In[94]:


df_new


# Creo un HeatMap para entender mejor la información de mi df.

# In[95]:


import matplotlib.pyplot as plt


# In[96]:


corr=df_new.corr()
plt.figure(figsize=(10,8)) #grafico mas grande
sns.heatmap(corr,annot=True,fmt=".2f");  #info en dos decimales


# In[97]:


sns.lmplot(data=df_new, x="Human Development Index",y="Adolescent Birth Rate",hue="Continent", line_kws= {'color':'red'})


# In[98]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(data=df_new, x="Human Development Index",y="Adolescent Birth Rate", line_kws= {'color':'red'})

#anadir una dimension mas con region/continente para distinguir y facilitar visualizacion


# In[99]:


#plt.figure(figsize=(15,38))
#sns.barplot(data=df_new, x= 'mmr_', y ='continent',hue='hdi_')


# In[100]:


df_new.groupby('Country')['Human Development Index'].mean().sort_values(ascending=False).head(5)


# In[101]:


df_new.groupby('Country')['Human Development Index'].mean().sort_values(ascending=True).head(5).drop('Somalia')


# In[102]:


# Your existing code
result = df_new.groupby('Country')['Human Development Index'].mean().sort_values(ascending=True).head(5).drop('Somalia')

# Extract country names
top_countries_list = result.index.tolist()

# Print the list
print(top_countries_list)


# In[103]:


# Your existing code
result = df_new.groupby('Country')['Human Development Index'].mean().sort_values(ascending=False).head(5)

# Extract country names
top_countries_list2 = result.index.tolist()

# Print the list
print(top_countries_list2)


# In[104]:


df_top_countries = df_new[df_new['Country'].isin(top_countries_list) | df_new['Country'].isin(top_countries_list2)]


# In[105]:


plt.figure(figsize=(15,15))
sns.lineplot(data=df_top_countries, x= 'Year', y= 'Human Development Index',hue= 'Country',err_kws = None)

#buscara en chat gpt como hacer grafico solo con los 5 maiores y 5 menores, incluido os 10 numa lista


# In[106]:


sns.regplot(data=df_new, x="Human Development Index",y="Adolescent Birth Rate",line_kws= {'color':'red'})


# In[107]:


sns.regplot(data=df_top_countries, x="Human Development Index",y="Adolescent Birth Rate",line_kws= {'color':'red'})


# In[108]:


import plotly.express as px

# Create the line plot
fig = px.line(df_top_countries, x='Year', y='Human Development Index', color='Country')
fig.show()


# In[109]:


px.strip(df_new, x="Maternal Mortality Rate")


# In[110]:


px.strip(df_new, x="Human Development Index", color="Continent", hover_name="Country")

#delimitar por años


# In[111]:


# Supongamos que tu DataFrame tiene una columna llamada 'año'
# Puedes cambiar el nombre de la columna según tu conjunto de datos
fig = px.strip(df_new, x="Human Development Index", color="Continent", hover_name="Country", animation_frame="Year")

# Ajusta el diseño y muestra la figura
fig.update_layout(title="Desarrollo Humano de 2000 a 2021")
fig.show()



# In[112]:


px.histogram(df_new, x="Human Development Index", color="Continent", hover_name="Country")


# In[113]:


px.histogram(df_new, x="Adolescent Birth Rate", color="Continent", hover_name="Country", marginal="rug")


# In[114]:


df_europa = df_new[df_new.Continent=='Europa']


# In[115]:


px.histogram(df_europa, y="Life Expectancy at Birth", x="Country", hover_name="Country", marginal="rug",histfunc="avg")


# In[116]:


sns.displot(df_new, x = "Continent" , y = "Human Development Index")

plt.xticks(rotation=90)


# In[117]:


px.histogram(df_new, x="Life Expectancy at Birth", y="Country", color="Continent", hover_name="Country", marginal="rug")

#puedo seleccionar los paises con menor y mayor indice y trazar paralelos"


# In[118]:


px.histogram(df_new, x="Female Share of Seats in Parliament", y="Country", color="Continent", hover_name="Country", marginal="rug")


# In[119]:


px.histogram(df_new, x="Female Share of Seats in Parliament", y="Human Development Index", color="Continent", hover_name="Country", marginal="rug", facet_col="Continent")


# Columns : 'Country', 'Continent', 'Year', 'Adolescent Birth Rate', 'co_prod_', 'Expected Years of Schooling', 'Expected Years of Schooling for Females','Gross National Income per capita', 'Human Development Index', 'Life Expectancy at Birth', 'Females Life Expectancy at Birth', 'Maternal Mortality Rate', 'Mean Years of Schooling', 'Participation Rate of Females in the Labor Force', 'Female Share of Seats in Parliament']

# In[120]:


px.histogram(df_new, x="Human Development Index", y="Female Share of Seats in Parliament", color="Continent", hover_name="Country", marginal="rug", facet_col="Continent")


# In[121]:


px.bar(df_new, color="Gross National Income per capita", x="Human Development Index", y="Continent", hover_name="Country" )

#por que no funciona?


# In[122]:


px.sunburst(df_new, color="co_prod_", values="Human Development Index", path=["Continent","Country"], hover_name="Country" )


# In[123]:


px.sunburst(df_new, color="Gross National Income per capita", values="Human Development Index", path=["Continent","Country"], hover_name="Country" )


# In[124]:


px.treemap(df_new, color="Participation Rate of Females in the Labor Force", values="Human Development Index", path=["Continent","Country"], hover_name="Country",height=600 )


# In[125]:


px.choropleth(df_new, color="Human Development Index", locations="Continent", hover_name="Continent",height=400 )


# In[126]:


px.scatter(df_new, x="Mean Years of Schooling", y="Human Development Index", hover_name="Country", color="Continent")


# In[127]:


df_new.isnull().sum()


# In[128]:


px.scatter(df_new, x="Mean Years of Schooling", y="Human Development Index", hover_name="Country", color="Continent", size="Gross National Income per capita" )


# In[129]:


px.scatter(df_new, y="Mean Years of Schooling", x="Human Development Index", hover_name="Country", color="Continent", size="Gross National Income per capita" )


# In[130]:


px.scatter(df_europa, y="Mean Years of Schooling", x="Human Development Index", hover_name="Country", color="Country", size="Gross National Income per capita" )


# In[131]:


px.scatter(df_europa, y="Mean Years of Schooling", x="Year", hover_name="Country", color="Country", size="Gross National Income per capita" )


# In[132]:


px.scatter(df_europa, y="Female Share of Seats in Parliament", x="Human Development Index", hover_name="Country", color="Country", size="Gross National Income per capita" )


# In[133]:


px.scatter(df_europa, y="Female Share of Seats in Parliament", x="Year", hover_name="Country", color="Country", size="Gross National Income per capita" )


# Pre procesado para Clustering
# 
# Empiezo el pre procesado para el clustering, con LAbel Encoder pra transformar a numericos

# In[134]:


from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[135]:


label = LabelEncoder()


# In[136]:


df_trans = df_new.copy()


# In[137]:


df_trans.Country = label.fit_transform(df_trans.Country)


# In[138]:


label.classes_


# In[139]:


df_trans.head()


# In[140]:


df_trans.Continent = label.fit_transform(df_trans.Continent)


# In[141]:


standard = StandardScaler()


# In[142]:


df_trans.columns


# In[143]:


numeric = [ 'Adolescent Birth Rate', 'co_prod_',
       'Expected Years of Schooling',
       'Expected Years of Schooling for Females',
       'Gross National Income per capita', 'Human Development Index',
       'Life Expectancy at Birth', 'Females Life Expectancy at Birth',
       'Maternal Mortality Rate', 'Mean Years of Schooling',
       'Participation Rate of Females in the Labor Force',
       'Female Share of Seats in Parliament']


# In[144]:


df_trans[numeric]= standard.fit_transform(df_trans[numeric])


# In[145]:


df_trans.head()


# In[7]:


import seaborn as sns; 
sns.set(rc = {"scatter.edgecolors":"k", 'figure.figsize': (15, 9)}) 


# Modelo: 
# empiezo con KMeans para crear los clusters 
# 
# 
# 

# In[147]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=4)
#lo entrenamos 
kmeans.fit(df_trans)
y_kmeans = kmeans.predict(df_trans)


# In[148]:


df_new['cluster'] = y_kmeans


# In[149]:


df_new.head()


# In[150]:


df_new.cluster.value_counts()


# In[151]:


g = sns.boxplot(data = df_new, x = 'Expected Years of Schooling')

# Add a title and change xlabel
g.set_title('Box Plot of Total')
g.set_xlabel('Expected Years of Schooling')


# In[152]:


sns.displot(df_new, x = "Continent" , y = "Human Development Index", hue = "cluster")

plt.xticks(rotation=90)


# In[153]:


sns.displot(df_new, x = "Continent" , y = "co_prod_", hue = "cluster")

plt.xticks(rotation=90)


# In[154]:


sns.displot(df_new, x = "Continent" , y = "Adolescent Birth Rate", hue = "cluster")

plt.xticks(rotation=90)


# In[155]:


sns.displot(df_new, x = "Continent" , y = "Maternal Mortality Rate", hue = "cluster")

plt.xticks(rotation=90)


# In[156]:


sns.displot(df_new, x = "Continent" , y = "Life Expectancy at Birth", hue = "cluster")

plt.xticks(rotation=90)


# In[157]:


sns.displot(df_new, x = "Continent" , y = "Females Life Expectancy at Birth", hue = "cluster")

plt.xticks(rotation=90)


# In[158]:


sns.displot(df_new, x = "Continent" , y = "Participation Rate of Females in the Labor Force", hue = "cluster")

plt.xticks(rotation=90)


# In[159]:


sns.displot(df_new, x = "Continent" , y = "Female Share of Seats in Parliament", hue = "cluster")

plt.xticks(rotation=90)


# In[160]:


sns.displot(df_new, x = "Continent" , y = "Expected Years of Schooling", hue = "cluster")

plt.xticks(rotation=90)


# In[161]:


sns.displot(df_new, x = "Continent" , y = "Expected Years of Schooling for Females", hue = "cluster")

plt.xticks(rotation=90)


# In[162]:


sns.barplot(data = df_new, x = "Continent" , y = "Human Development Index" , hue = "cluster")

plt.xticks(rotation=90)


# In[163]:


sns.barplot(data = df_new, x = "Continent" , y = 'co_prod_' , hue = "cluster")

plt.xticks(rotation=90)


# In[164]:


sns.barplot(data = df_new, x = "Continent" , y = "Adolescent Birth Rate" , hue = "cluster")

plt.xticks(rotation=90)


# In[165]:


sns.barplot(data = df_new, x = "Continent" , y = "Maternal Mortality Rate" , hue = "cluster")

plt.xticks(rotation=90)


# In[166]:


sns.barplot(data = df_new, x = "Continent" , y = 'Life Expectancy at Birth' , hue = "cluster")

plt.xticks(rotation=90)


# In[167]:


sns.barplot(data = df_new, x = "Continent" , y = "Participation Rate of Females in the Labor Force" , hue = "cluster")

plt.xticks(rotation=90)


# In[168]:


sns.barplot(data = df_new, x = "Continent" , y = 'Female Share of Seats in Parliament' , hue = "cluster")

plt.xticks(rotation=90)


# In[169]:


sns.barplot(data = df_new, x = "Continent" , y = 'Expected Years of Schooling for Females' , hue = "cluster")

plt.xticks(rotation=90)


# In[170]:


#ejemplo Alana

fig, axes = plt.subplots(11, 3, figsize=(15, 40))

for i, variable in enumerate(numeric):
    for j, cluster in enumerate(df_new['cluster'].unique()):
        cluster_data = df_new[df_new['cluster'] == cluster]
        if i < 11 and j < 3: 
            sns.histplot(data=cluster_data, x=variable, kde=True, label=f'Cluster {cluster}', ax=axes[i, j])
            #px.histogram(data_frame=df_new, x = variable, marginal= "kde",)
            
            axes[i, j].set_ylabel('count')
            axes[i, j].set_title(variable)
            axes[i, j].legend()
            
        
plt.tight_layout()


# In[171]:


sns.countplot(df_new.Continent, hue = df_new.cluster)


# Pre procesado para supervisado

# In[172]:


from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[173]:


label = LabelEncoder()


# In[174]:


df_sup = df_new.copy()


# In[175]:


df_sup.Country = label.fit_transform(df_sup.Country)


# In[176]:


label.classes_


# In[177]:


df_sup.head()


# In[178]:


df_sup.Continent = label.fit_transform(df_sup.Continent)


# In[179]:


standard = StandardScaler()


# In[180]:


df_sup.columns


# In[181]:


numeric = [ 'Adolescent Birth Rate', 'co_prod_',
       'Expected Years of Schooling',
       'Expected Years of Schooling for Females',
       'Gross National Income per capita',
       'Life Expectancy at Birth', 'Females Life Expectancy at Birth',
       'Maternal Mortality Rate', 'Mean Years of Schooling',
       'Participation Rate of Females in the Labor Force',
       'Female Share of Seats in Parliament']


# In[182]:


df_sup[numeric]= standard.fit_transform(df_sup[numeric])


# In[183]:


df_sup.head()


# In[184]:


df_sup.drop('cluster',axis=1, inplace=True)


# In[185]:


import seaborn as sns; 
sns.set(rc = {"scatter.edgecolors":"k", 'figure.figsize': (15, 9)}) 


# Outliers con Z-score

# In[186]:


# Import zscore function
from scipy.stats import zscore

# Calculate z-score for each data point and compute its absolute value
z_scores = zscore(df_new['Expected Years of Schooling'])
abs_z_scores = np.abs(z_scores)

# Select the outliers using a threshold of 3
outliers = df_new[abs_z_scores > 3]
outliers.head()


# Evaluar :: Elegir numero de clusters utilizando elbow method

# In[187]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Select the numeric variables for PCA
data_for_pca = df_new[numeric]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_pca)

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Add the PCA components to the DataFrame
df_new['PCA1'] = pca_result[:, 0]
df_new['PCA2'] = pca_result[:, 1]

# Range of clusters to consider
k_values = range(1, 11)

# Initialize an empty list to store inertia values
inertia_values = []

# Iterate over different values of k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_new[['PCA1', 'PCA2']])
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(12, 6))

# Plotting the elbow curve
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')

# Plotting the scatter plot with optimal k
plt.subplot(1, 2, 2)
optimal_k = 3  # Adjust based on the visual inspection of the elbow plot
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
df_new['Cluster'] = kmeans_optimal.fit_predict(df_new[['PCA1', 'PCA2']])
sns.scatterplot(data=df_new, x='PCA1', y='PCA2', hue='Cluster', palette='viridis')
plt.title(f'Clustering with K={optimal_k}')

plt.tight_layout()
plt.show()


# In[188]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming df_new is your DataFrame and numeric is the list of numeric variables

# Set up the subplots
fig, axes = plt.subplots(12, 3, figsize=(25, 70))

# Variable index starts from 1
for i, variable in enumerate(numeric, start=1):
    # Cluster index starts from 0
    unique_clusters = df_new['cluster'].nunique()
    for j, k in enumerate(range(2, min(unique_clusters + 2, 8))):  # Adjust the range as needed
        # Fit KMeans with different cluster numbers
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_new[[variable]])
        
        # Plot histogram
        sns.histplot(data=df_new, x=variable, kde=True, hue=kmeans.labels_, multiple="stack", ax=axes[i, j])
        axes[i, j].set_ylabel('count')
        axes[i, j].set_title(f'{variable} - K={k}')
        axes[i, j].legend()
        
        # Display the inertia value
        inertia_value = kmeans.inertia_
        axes[i, j].text(0.5, -0.1, f'Inertia: {inertia_value:.2f}', size=10, ha="center", transform=axes[i, j].transAxes)

plt.tight_layout()
plt.show()


# In[189]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Assuming df_new is your DataFrame and numeric is the list of numeric variables

# Set up the subplots
fig, axes = plt.subplots(11, 3, figsize=(15, 58))

# Variable index starts from 1
for i, variable in enumerate(numeric, start=1):
    # Cluster index starts from 0
    unique_clusters = df_new['cluster'].nunique()
    for j, k in enumerate(range(2, min(unique_clusters + 2, 8))):  # Adjust the range as needed
        # Fit KMeans with different cluster numbers
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_new[[variable]])
        
        # Plot histogram
        sns.histplot(data=df_new, x=variable, kde=True, hue=kmeans.labels_, multiple="stack", ax=axes[i, j])
        axes[i, j].set_ylabel('count')
        axes[i, j].set_title(f'{variable} - K={k}')
        axes[i, j].legend()
        
        # Display the inertia value
        inertia_value = kmeans.inertia_
        axes[i, j].text(0.5, -0.1, f'Inertia: {inertia_value:.2f}', size=10, ha="center", transform=axes[i, j].transAxes)

plt.tight_layout()
plt.show()


# In[ ]:


df_new.describe()


# In[ ]:


g = sns.histplot(data = df_new, x = 'Expected Years of Schooling')
# Add labels
g.set_xlabel('Expected Years of Schooling')


# In[ ]:


import plotly.express as px

# Assuming df_new is your DataFrame
fig = px.histogram(df_new, x='Expected Years of Schooling', nbins=20, labels={'Expected Years of Schooling': 'Expected Years of Schooling'})

# Update the layout to include hover information
fig.update_layout(hovermode='x unified')

# Show the plot
fig.show()


# In[ ]:


#from sklearn.metrics import silhouette_score
#silhouette_avg = silhouette_score(data, labels)


# Modelo de predicción

# In[191]:


df_sup.head(20)


# In[192]:


X = df_sup.drop("Human Development Index",axis = 1)

y = df_sup["Human Development Index"]


# In[193]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split( X,y,train_size = 0.8, random_state = 4)


# In[194]:


X_train.shape


# In[195]:


X_test.shape


# In[196]:


#usamos modelo regresor porque el HDI es un numero continuo

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[197]:


linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)


# In[203]:


sns.kdeplot(y_test, label="y_test")

sns.kdeplot( y_pred_linear, label="LinearRegression")

plt.legend()


# In[213]:


from sklearn.metrics import mean_squared_error, r2_score
MSEL=mean_squared_error(y_test, y_pred_linear)
print("MSE",MSEL)
R2L= r2_score(y_test, y_pred_linear)
print("R2",R2L)


# In[205]:


random = RandomForestRegressor()

random.fit(X_train, y_train)

y_pred_random = random.predict(X_test)


# In[210]:


MSER = mean_squared_error(y_test, y_pred_random)
print("MSE",MSER)
R2R = r2_score(y_test, y_pred_random)
print("R2",R2R)


# In[211]:


tree = DecisionTreeRegressor()

tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
MSET=mean_squared_error(y_test, y_pred_tree)
print("MSE",MSET)
R2T=r2_score(y_test, y_pred_tree)
print("R2",R2T)


# In[212]:


knn = KNeighborsRegressor()

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
MSEK=mean_squared_error(y_test, y_pred_knn)
print("MSE")
R2K= r2_score(y_test, y_pred_knn)
print("R2")


# In[209]:


sns.kdeplot(y_test, label="y_test")

sns.kdeplot( y_pred_linear, label="LinearRegression")
sns.kdeplot( y_pred_random, label="RandomForest")
sns.kdeplot( y_pred_tree, label="DecisionTree")
sns.kdeplot( y_pred_knn, label="KNeighbors")


plt.legend()


# In[215]:


metrics = {"modelo":["RegressorLineal","DecisionTree","RandomForest","KNeighbor"],
           "MSE":[MSEL,MSET,MSER,MSEK],
           "R2": [R2L,R2T,R2R,R2K]}

df_metrics = pd.DataFrame(metrics)

df_metrics

