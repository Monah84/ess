import extended_scispacy as es
import pandas as pd

def generate_example():
    dt = pd.DataFrame([
    'patient complained from chest pain.',
    'patient suffers from headache on off for almost 2month.pt has uncontrolled dm htn and dyslipidemia .needs medication adjustment.. hypothyrodism dyslipidemia',
    'dm hyperlipidemia.his ck was found markedly elevated .all investigations came negative no signe for cancers or muscular dystrophies were founded. ctscan of abd only showed egenerative changes in l5-s1. needs neurological cosultation.',
    'patient suffering from burning feet body weakness'])
    dt.columns = ['text']
    return dt

if __name__=="__main__":
    data  = generate_example()
    cleansing = es.SciSpacyCleansing(apply_rules=True,delimiter=' ')
    result  = cleansing.run(data,column_name='text',processed_column_name='cleaned')