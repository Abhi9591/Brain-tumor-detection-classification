import streamlit as st

import base64
import cv2

# ================ Background image ===

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Brain Tumour Detection Using CNN-2D"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('....CODE/mainproject/1.avif')


st.title("Welcome Login Page !!!")

import pandas as pd

# df = pd.read_csv('login_record.csv')

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

col1, col2 = st.columns(2)

     
with col1:

    UR1 = st.text_input("Login User Name",key="username")
    psslog = st.text_input("Password",key="password",type="password")
    # tokenn=st.text_input("Enter Access Key",key="Access")
    agree = st.checkbox('LOGIN')
    
    if agree:
        try:
            
            df = pd.read_csv(UR1+'.csv')
            U_P1 = df['User'][0]
            U_P2 = df['Password'][0]
            if str(UR1) == str(U_P1) and str(psslog) == str(U_P2):
                st.success('Successfully Login !!!')    

        
                import pandas as pd
                 
                def hyperlink(url):
                     return f'<a target="_blank" href="{url}">{url}</a>'
                 
                dff = pd.DataFrame(columns=['page'])
                dff['page'] = ['MainFile']
                dff['page'] = dff['page'].apply(hyperlink)
                dff = dff.to_html(escape=False)
 
                st.write(dff, unsafe_allow_html=True)   

            else:
                st.write('Login Failed!!!')
        except:
            st.write('Login Failed!!!')                 
with col2:
    UR = st.text_input("Register User Name",key="username1")
    pss1 = st.text_input("First Password",key="password1",type="password")
    pss2 = st.text_input("Confirm Password",key="password2",type="password")
    # temp_user=[]
        
    # temp_user.append(UR)
    
    if pss1 == pss2 and len(str(pss1)) > 2:
        import pandas as pd
        
  
        import csv 
        
        # field names 
        fields = ['User', 'Password'] 
        

        
        # st.text(temp_user)
        old_row = [[UR,pss1]]
        
        # writing to csv file 
        with open(UR+'.csv', 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow(fields) 
                
            # writing the data rows 
            csvwriter.writerows(old_row)
        st.success('Successfully Registered !!!')
    else:
        
        st.write('Registeration Failed !!!') 
        
        
        