import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

def load_model(filename):
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model

# Load the trained XGBoost model
xgbr = load_model("xgboost_regressor_model.pkl")

with st.sidebar:
        st.image(r".\assets\HousePrice.png", use_column_width=True)
        st.write("Select The Options Below")
        baths = int(st.slider('Bath Rooms', 0, 4, 2))
        balcony = int(st.slider('Balcony', 0, 4, 2))
        no_of_rooms = int(st.slider('No of Rooms', 1, 5, 2))

        flat_type_map = {'BHK': 0, 'R': 1, 'RK': 2, 'BH': 3}
        flat_type = flat_type_map[st.selectbox('Flat Type', flat_type_map.keys())]

        city_map = {'Chennai': 0, 'Bangalore': 1, 'Hyderabad': 2, 'Mumbai': 3, 'Thane': 4, 'Kolkata': 5, 'Pune': 6, 'Delhi': 7}
        city = city_map[st.selectbox('City', city_map.keys())]

        total_area_scaled = st.slider('Total Area', 0, 2060, 1000,step=250) / 2060

        st.write('Developed by [Darshan Pathak](https://www.linkedin.com/in/pathakdarshan12)')
        st.markdown('This Application Uses Linear Regression for Prediction')
        st.markdown('**App Framework** - **Streamlit**')

        input_data = pd.DataFrame([[baths, balcony, no_of_rooms, flat_type, city, total_area_scaled]],
                          columns=['baths', 'balcony', 'no_of_rooms', 'flat_type', 'city', 'total_area_scaled'])

        if st.button('How XGBoostRegressor Works'):
            st.markdown("""
                ## How XGBoostRegressor Works:
    
                While Buying a House we try to estimate how much a house is worth considering various factors like size, location, total area, and number of rooms.
    
                **Traditional approach:** You might weigh each factor based on experience and guesstimate the price. This can be imprecise and miss hidden patterns.
    
                **XGBoostRegressor approach:**
    
                - It breaks down the problem into smaller questions, like "how much does size affect price?" or "how does location affect price?".
                - It answers these questions using simple models, like "houses with more bedrooms are generally more expensive".
                - It combines these answers, considering how they interact with each other. For example, "a large house in a good location might be even more expensive".
                - It does this repeatedly, learning from each step to make a more accurate final prediction.
    
                This way, XGBoostRegressor considers all factors, learns from its "experience", and avoids getting stuck on just one aspect, leading to more accurate house price predictions!
            """)

            st.write() 

            st.markdown('To know more, check out my Jupyter Notebook [here](https://link-to-your-notebook).')

        
    
def main():
    start_image = st.image("https://storage.googleapis.com/kaggle-media/competitions/House%20Prices/kaggle_5407_media_housesbanner.png", use_column_width=True)
    image_link="https://i.imgur.com/Q5IhUpF.gif"
    df = pd.read_csv("HousePrice.csv")
    global fig
    fig = None
    st.title("House Price Prediction App")
    st.markdown("""
        ##### A web application for predicting Indian Housing Prices.

        This app uses machine learning to predict the price of the house. 
        It loads a pre-trained linear regression model, which takes as input various features of the house, 
        such as the number of rooms, the number of bedrooms, the total area, and the type of the flat. 
        The app preprocesses the input data by combining some of the features.
        """)
    st.markdown("**🔗 [GitHub repository](https://github.com/Pathakdarshan12)** | 💼 **LinkedIn profile:** [@pathakdarshan12](https://www.linkedin.com/in/pathakdarshan12)")
    
    st.write('Please Enter the details in the sidebar to know the results')

    if st.button('Predict House Price'):
        price_prediction = xgbr.predict(input_data)
        price = int(np.round((price_prediction[0]* 18504560)/100000, -1))
        st.metric(label='Median House Value', value=f"₹ {price:.2f} Lakhs")
 
        city_names = ['Chennai', 'Bangalore', 'Hyderabad', 'Mumbai', 'Thane', 'Kolkata', 'Pune', 'Delhi']
        image_link = f"assets/{city_names[city]}.jpg"
        
        
        mean_house_prices = []
        for i in range(len(city_map)):
            city_data = df[df['city'] == i]
            mean_house_price = city_data['price_scaled'].mean()*185.0456
            mean_house_prices.append(mean_house_price)

        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'red']

        fig = go.Figure(data=[
            go.Bar(x=list(city_map.keys()), y=mean_house_prices, marker_color=colors)
        ])

        mean_prices = []
        median_prices = []
        mode_prices = []

        for i in range(len(city_map)):
            city_data = df[df['city'] == i]
            mean_prices.append(city_data['price_scaled'].mean()*185.0456)
            median_prices.append(city_data['price_scaled'].median()*185.0456)
            mode_price = city_data['price_scaled'].mode()*185.0456
            if len(mode_price) > 0:
                mode_prices.append(mode_price[0])
            else:
                mode_prices.append(np.nan)

        for i, mean_house_price in enumerate(mean_house_prices):
            fig.add_annotation(
                x=list(city_map.keys())[i],
                y=mean_house_price,
                text=f"Avg: {mean_prices[i]:.2f}<br>Med: {median_prices[i]:.2f}<br>Mode: {mode_prices[i]:.2f}",
                showarrow=False,
                font=dict(color='white')
            )

        fig.update_layout(
            title='Mean House Price for Each City in Lakhs',
            xaxis=dict(title='City', color='white', showgrid=False),
            yaxis=dict(title='Mean House Price', color='white', showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='red'),
            margin=dict(t=50, l=50, r=50, b=50),
        )
        st.plotly_chart(fig)
        
    main_image = st.image(image_link, use_column_width=True)
    

if __name__ == '__main__':
    main()
