import pandas as pd

df = pd.read_excel('../Barridos.xlsx', sheet_name='Hoja1')
print(df.head())

prueba = ['hola', 'que', 'tal', 'estas', 'yo', 'bastante', 'cansado']
df = pd.DataFrame(prueba)
print(df)

with pd.ExcelWriter('../Barridos.xlsx', mode='a', if_sheet_exists='overlay') as file:
    df.to_excel(file, sheet_name='Hoja1', index=False)
    print('writen')