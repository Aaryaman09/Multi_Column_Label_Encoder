import os,pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LABELENCODER_OBJECT_DIR = "LabelEncoder_Object_Dir"
PICKLE_FILE_NAME = "LabelEncoderObject.pkl"

class MultiColumnLabelEncoder:

    def __init__(self,dataframe,dataframe_name="Sample"):
        self.df = dataframe # array of column names to encode
        self.columns = [column_name for column_name in self.df.columns if self.df[column_name].dtypes == "object"]
        self.dataframe_dir_name = dataframe_name
        self.encoders = {}
        self.mapping = {}
        os.makedirs(LABELENCODER_OBJECT_DIR,exist_ok=True)


    def fit(self):
        for col in self.columns:
            self.encoders[col] = LabelEncoder().fit(self.df[col])
            arr = self.encoders[col].classes_

            self.mapping[col] = {}
            for ind,ele in enumerate(arr):
                self.mapping[col][str(ind)] = ele

        print(f"Mapping Created : {self.mapping}")
        self.save()
        return self


    def save(self):
        MAPPING_DIR = os.path.join(LABELENCODER_OBJECT_DIR,self.dataframe_dir_name)
        os.makedirs(MAPPING_DIR,exist_ok=True)

        for col in self.mapping.keys():
            column_name_dir = os.path.join(MAPPING_DIR,col)
            os.makedirs(column_name_dir,exist_ok=True)

            pickle_file_path = os.path.join(column_name_dir,PICKLE_FILE_NAME)

            out_file = open(pickle_file_path, "wb")
            pickle.dump(self.encoders[col], out_file)
            out_file.close()
        
        return "Encoding exported."


    def transform(self):
        output = self.df.copy()
        for col in self.columns:
            output[col] = self.encoders[col].transform(self.df[col])
        return output


    def fit_transform(self):
        return self.fit().transform()


    def load(self):
        MAPPING_DIR = os.path.join(LABELENCODER_OBJECT_DIR, self.dataframe_dir_name)
        loaded_labelencoders = {}

        column_names = os.listdir(MAPPING_DIR)
        for key in column_names:
            column_name_dir = os.path.join(MAPPING_DIR,key)

            pickle_file_path = os.path.join(column_name_dir,PICKLE_FILE_NAME)
            in_file = open(pickle_file_path, "rb")
            loaded_labelencoders[key] = pickle.load(in_file)
            in_file.close()
        
        return loaded_labelencoders    


    def inverse_transform(self):
        loaded_labelencoders = self.load()
        output = self.df.copy()
        columns = [column_name for column_name in loaded_labelencoders.keys()]
        for col in columns:
            output[col] = loaded_labelencoders[col].inverse_transform(self.df[col])
        return output



if __name__=="__main__":

    df = pd.DataFrame({ 'city':     ['London','Paris','Moscow'],
                        'size':     ['M',     'M',    'L'],
                        'quantity': [12,       1,      4]})

    print(f"Original DataFrame : \n\n{df}\n\n")
    multi = MultiColumnLabelEncoder(df,"City_analysis")
    X = multi.fit_transform()
    print(f"After Transformation : \n\n{X}\n\n")
    
    df_1 = pd.DataFrame({   'city':     [2,   1,    0],
                            'size':     [1,   0,    0],
                            'quantity': [12,  1,    4]})
    

    print(f"Testing New Dataset : \n\n{df_1}\n\n")   
    multi = MultiColumnLabelEncoder(df_1,"City_analysis")                    
    inv = multi.inverse_transform()
    print(f"After Transformation : \n\n{inv}\n\n")

