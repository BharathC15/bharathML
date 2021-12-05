import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
np.random.seed(123)

fig = make_subplots(specs=[[{'secondary_y': True}]])

n=8

d = {'date': [datetime(2020, 3, k ) for k in range(2, n+2)],
     'A': 21+12*np.random.rand(n),
     'B': 16+12*np.random.rand(n),
     'C': 14+10*np.random.rand(n)}
df = pd.DataFrame(d)

fig.add_trace(go.Scatter(x=df['date'], y= df['A']), secondary_y=False)  
fig.add_trace(go.Scatter(x=df['date'], y= df['B']), secondary_y=False)  

fig.add_trace(go.Scatter(x=df['date'], y= df['C']), secondary_y=True) 
fig.show()
