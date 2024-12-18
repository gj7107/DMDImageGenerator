import dash_ag_grid as dag
from dash import Dash, html, dcc, Input, Output, callback, Patch, State, no_update
import dash
import pandas as pd
import dash_bootstrap_components as dbc
import imgtool
import feffery_antd_components as fac
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

"""
written_text = ''
J = np.zeros(1)
img = np.zeros(1)

D = 0
W = 0
Neibhor_TH = 0
AtomPlane_size = [0,0]
image_size = [0,0]

ang1 = 0
ang2 = 0
L1 = 0
L2 = 0
offset = [0,0]
FLAG_STICK_NEIGHBOR = False

ring_start = 0
ring_end = 0
ring_offset = [0,0]
ring_strength = 0

tilt_angle = 0
tilt_strength = 0

addoffset_value = 0

polynomial_order = 0
polynomial_mode = 0
polynomial_range = 0
polynomial_offset = [0,0]

circle_radius = 0
circle_setvalue = 0
FLAG_CIRCLE_INTERIOR = False
circle_offset = [0,0]

th_lower = 0
th_lowerset = 0
th_upper = 1
th_upperset = 1

Multi = 1
savefilename = ""
"""

initial_values = {
    # text
    'written_text':"for i in range(J.shape[0]) : \n    for j in range(J.shape[1]) : \n        J[i,j]=np.mod((i+j),2)",
    # Transform
    'ang1':-7.5,
    'ang2':82.5,
    'L1':16.5,    
    'L2':16.5,    
    'offset':[0,0],    
    # Mutliplier
    'Multi':1,
    'W':6,
    'D':8.25,
    'Neighbor_TH':0.8,
    'AtomPlane_size':[51,51],
    'image_size':[768,1024],
    'FLAG_STICK_NEIGHBOR':True,
    # Threshold panel
    'th_lower':0,
    'th_lowerset':0,
    "th_upper":1,
    "th_upperset":1,
    # Ring pattern
    'ring_start':100,
    'ring_end':250,
    'ring_offset':[0,0],
    'ring_strength':1,
    # tilt pattern
    'tilt_angle' : -7.5,
    'tilt_strength' : 1,
    # offset pattern
    'addoffset_value' : 0,
    # polynomial pattern    
    'polynomial_order' : 2,
    'polynomial_mode' : 1,
    'polynomial_range' : 200,
    'polynomial_offset' : [0,0],
    # circle pattern
    'circle_radius' : 200,
    'circle_setvalue' : 0,
    'FLAG_CIRCLE_INTERIOR' : True,
    'circle_offset' : [0,0],
    # ETC
    'Multi' : 1, 
    'savefilename' : ""
}


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks=True)
#def GenerateImage(J, T, W=6, Neighbor_TH = 0.8,D=8.25, AtomPlaneSize = [51, 51], imsize= [768,1024], FLAG_STICK_NEIGHBOR = True)

heading = html.H1("Generate DMD patterns",className="bg-secondary text-white p-2 mb-4")

MainView = html.Div([
    dbc.Row(fac.AntdSpace(
            [
                html.Div( [ 
                dbc.Button("New", id="button-new-image", color = 'primary', style={"width":"20%"})]),
                html.Div([
                dbc.Button("Line X", id="button-lineX-code",  color = 'secondary', style={"width":"20%"}),
                dbc.Button("Line Y", id="button-lineY-code", color = 'secondary',  style={"width":"20%"}),
                dbc.Button("Stripe X", id="button-stripeX-code", color = 'light',  style={"width":"20%"}),
                dbc.Button("Stripe Y", id="button-stripeY-code", color = 'light',  style={"width":"20%"}),],
                 style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),
                fac.AntdInput(
                    id='text-area-demo',
                    placeholder='Please enter content...',
                    mode='text-area',
                    style={
                        'width': 400
                    },
                    autoSize={
                        'minRows': 3,
                        'maxRows': 6
                    },
                    value = "for i in range(J.shape[0]) : \n    for j in range(J.shape[1]) : \n        J[i,j]=np.mod((i+j),2)"                    
                ),
                html.Div("Set J values to a generate image."),
                dbc.Col([dbc.Button('Draw!', id = 'button-code-draw'),dbc.Button('Draw 10!', id = 'button-code-draw-rep')],width=3),      
                html.Div(
                    id='written-content',
                    style={
                        'whiteSpace': 'pre'
                    }
                )
            ],
            direction='vertical'
        )),
            ],
)


Transform = html.Div([
                html.Div( [ dbc.Label("Angle 1 (deg)",style={"margin-light":"20px"}),
                dbc.Input(id="input-ang1", type="text", value = "-7.5", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("Angle 2 (deg)",style={"margin-light":"20px"}),
                dbc.Input(id="input-ang2", type="text", value = "82.5", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("L1 (px)",style={"margin-light":"20px"}),
                dbc.Input(id="input-L1", type="text", value = "16.5", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("L2 (px)",style={"margin-light":"20px"}),
                dbc.Input(id="input-L2", type="text", value = "16.5", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("Offset (px,px)",style={"margin-light":"20px"}),
                dbc.Input(id="input-offset", type="text", value = "0,0", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
])
Params = html.Div([
    html.Div( [ dbc.Label("Multiplier",style={"margin-light":"20px"}),
                dbc.Input(id="input-Multi", type="text", value = "1", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
    html.Div( [ dbc.Label("W",style={"margin-light":"20px"}),
                dbc.Input(id="input-W", type="text", value = "6", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
    html.Div( [ dbc.Label("D",style={"margin-light":"20px"}),
                dbc.Input(id="input-D", type="text", value = "8.25", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
    html.Div( [ dbc.Label("Neighbor Threshold",style={"margin-light":"20px"}),
                dbc.Input(id="input-Neighbor-TH", type="text", value = "0.8", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
    html.Div( [ dbc.Label("AtomPlane Size",style={"margin-light":"20px"}),
                dbc.Input(id="input-atomplane-size", type="text", value = "51,51", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
    html.Div( [ dbc.Label("Image Size",style={"margin-light":"20px"}),
                dbc.Input(id="input-image-size", type="text", value = "768,1024", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
    html.Div( [ dbc.Label("Stick Neighbor",style={"margin-light":"20px"}),
                dbc.Switch(id="switch-stick-neighbor", value=True, style={'width': '50%'}),
                html.Div(id='div-empty-switch')],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'})
    ]
)

control_panel = dbc.Card(
    dbc.CardBody(
        [Transform, Params]
    )
)
threhold_panel = dbc.Card(
    dbc.CardBody(
        [
            html.Div( [ dbc.Label("If pixel < ",style={"margin-light":"20px"}),
            dbc.Input(id="input-threshold-lower", type="text", value = "0", style={'width': '20%'}), 
            dbc.Label("Set  ",style={"margin-light":"20px"}),
            dbc.Input(id="input-threshold-lowerset", type="text", value = "0", style={'width': '20%'}), 
            ],
            style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
            html.Div( [ dbc.Label("If pixel > ",style={"margin-light":"20px"}),
            dbc.Input(id="input-threshold-upper", type="text", value = "1", style={'width': '20%'}), 
            dbc.Label("Set  ",style={"margin-light":"20px"}),
            dbc.Input(id="input-threshold-upperset", type="text", value = "1", style={'width': '20%'}), 
            ],
            style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
            dbc.Button("Apply!", id='button-threshold-apply', n_clicks=0)
        ]
    )
)
accordion = html.Div(
    [
        dcc.Markdown(id="title_tool", children=f'''### Tools '''),
        dbc.Accordion([
            dbc.AccordionItem([threhold_panel], title = "Apply threshold"),
            dbc.AccordionItem([control_panel], title = "Set Parameters"),
        ],start_collapsed=True),
        dcc.Markdown(id="title_accordion", children=f'''### Some basic patterns '''),
        dbc.Accordion([
            dbc.AccordionItem(
                [html.P("Ring pattern"),
                html.Div( [ dbc.Label("Start radius",style={"margin-light":"20px"}),
                dbc.Input(id="input-ring-start", type="text", value = "100", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("End radius",style={"margin-light":"20px"}),
                dbc.Input(id="input-ring-end", type="text", value = "250", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("Center offset ",style={"margin-light":"20px"}),
                dbc.Input(id="input-ring-offset", type="text", value = "0,0", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("Strength ",style={"margin-light":"20px"}),
                dbc.Input(id="input-ring-strength", type="text", value = "1.0", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                dbc.Button("Draw", id="button-ring-draw", n_clicks = 0)                 
                ], title="Ring pattern"
            ),
            dbc.AccordionItem(
                [html.P("Tilt pattern"),
                html.Div( [ dbc.Label("Tilt angle",style={"margin-light":"20px"}),
                dbc.Input(id="input-tilt-angle", type="text", value = "-7.5", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("Tilt strength (/px)",style={"margin-light":"20px"}),
                dbc.Input(id="input-tilt-strength", type="text", value = "1", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                dbc.Button("Draw", id="button-tilt-draw", n_clicks = 0)                 
                ], title="Tilt pattern"
            ),
            dbc.AccordionItem(
                [html.P("Add Offset"),
                html.Div( [ dbc.Label("Offset value",style={"margin-light":"20px"}),
                dbc.Input(id="input-addoffset-value", type="text", value = "0", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                dbc.Button("Add", id="button-addoffset-draw", n_clicks = 0)                 
                ], title="Add Offset"
            ),

            dbc.AccordionItem(
                [html.P("Polynomial potential"),
                html.Div( [ dbc.Label("order (n)",style={"margin-light":"20px"}),
                dbc.Input(id="input-polynomial-order", type="text", value = "2", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("Potential range (px)",style={"margin-light":"20px"}),
                dbc.Input(id="input-polynomial-range", type="text", value = "200", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("Center offset (px)",style={"margin-light":"20px"}),
                dbc.Input(id="input-polynomial-offset", type="text", value = "0,0", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                
                html.Div( [        
                        dbc.RadioItems(
                        options=[
                            {"label": "X", "value": 1},
                            {"label": "Y", "value": 2},
                            {"label": "radial", "value": 3},
                        ],
                        value=1,
                        id="radioitems-polynomial",
                    ),html.Div(id='div-polynomial-empty')],
                style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                dbc.Button("Draw", id="button-polynomial-draw", n_clicks = 0)                 
                ], title="Polynomial potential"
            ),
            dbc.AccordionItem(
                [html.P("Fill circle"),
                html.Div( [ dbc.Label("Circle radius",style={"margin-light":"20px"}),
                dbc.Input(id="input-circle-radius", type="text", value = "200", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("Circle set value",style={"margin-light":"20px"}),
                dbc.Input(id="input-circle-setvalue", type="text", value = "0", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [ dbc.Label("Circle offset ",style={"margin-light":"20px"}),
                dbc.Input(id="input-circle-offset", type="text", value = "0,0", style={'width': '50%'}), ],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                html.Div( [        
                        dbc.RadioItems(
                        options=[
                            {"label": "Interior", "value": 1},
                            {"label": "Exterior", "value": 2},
                        ],
                        value=1,
                        id="radioitems-int-ext",
                    ),html.Div(id='div-fill-empty')],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                dbc.Button("Draw", id="button-circle-draw", n_clicks = 0)                 
                ], title="Fill pattern"
            ),
        ],
        start_collapsed=True,
    ),],
)


app.layout = dbc.Container(
    [
        dcc.Store(id="store-data", data=initial_values),
        dcc.Store(id="store-img", data={'img':np.zeros(1)}),
        heading,
        dbc.Row([
            dbc.Col([accordion], md=3),
            dbc.Col(
                [
                    dcc.Markdown(id="title", children=f'''## Image generation\n'''),
                    dbc.Row([dbc.Col(MainView),dbc.Col([html.Div( [ 
                html.H5(dbc.Badge("Save file name ",className="ms-1",style={"margin-light":"20px"})),
                dbc.Input(id="input-path-filename", type="text", value = r"C:\img.bmp", style={'width':'80%'})],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                                                        dcc.Graph(id="figure-area"),
                                                        dbc.Row(dbc.Button("Save", id='button-path-save')),
                                                        html.Div(id='div-path-empty')]),
                             ])
                ],  md=8,
            ),
        ]),
    ],
    fluid=True,
)



@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('written-content', 'children'),
    Input('text-area-demo', 'value'),
    State("store-data", 'data')
)
def sync_written_content(value, data):
    # global written_text
    written_text = value
    data['written_text'] = written_text
    return data, "Code : \n " + value

def GetNewImage(img, data) :
    image_size = data['image_size']
    img = np.zeros(image_size, dtype=np.float64)
    return img

def GetImageFromCode(img, data) :
    image_size = data['image_size']
    written_text = data['written_text']
    AtomPlane_size = data['AtomPlane_size']
    ang1 = data['ang1']
    ang2 = data['ang2']
    D = data['D']    
    W = data['W']
    offset = data['offset']
    L1 = data['L1']
    L2 = data['L2']
    Neighbor_TH = data['Neighbor_TH']
    FLAG_STICK_NEIGHBOR = data['FLAG_STICK_NEIGHBOR']
    try :
        print(written_text)
        print(AtomPlane_size)
        J = np.zeros(AtomPlane_size,dtype=np.float64)
        midpt_i = np.round(AtomPlane_size[0]/2) 
        midpt_j = np.round(AtomPlane_size[1]/2)        
        exec(written_text)
        print("execution success...")
        
        OffsetX = image_size[1]/2 - D*AtomPlane_size[0]/2 * (np.cos(ang1*np.pi/180) + np.sin(ang1*np.pi/180) ) + offset[0]
        OffsetY = image_size[0]/2 - D*AtomPlane_size[1]/2 * (np.cos(ang2*np.pi/180) + np.sin(ang2*np.pi/180) ) + offset[1]
        A = np.array([np.cos(ang1*np.pi/180)/40,np.sin(ang1*np.pi/180)/40,OffsetX])
        B = np.array([np.cos(ang2*np.pi/180)*L2/L1/40,np.sin(ang2*np.pi/180)*L2/L1/40,OffsetY])
        T = np.vstack((A.T,B.T))
        
        img = imgtool.GenerateImage(J, T, W=W, Neighbor_TH = Neighbor_TH, D=D, AtomPlaneSize = AtomPlane_size, imsize= image_size, FLAG_STICK_NEIGHBOR = FLAG_STICK_NEIGHBOR)
        img = img.astype(np.float64)
        print("Image generated...")
        #fig.update_layout(transition_duration=500)
        return img    
    except Exception as error:
        print("ERR", error)
        return -1

def GetRingImage(img, data) :
    image_size = data['image_size']
    ring_offset = data['ring_offset']
    ring_start = data['ring_start']
    ring_end = data['ring_end']
    ring_strength = data['ring_strength']
    
    img0 = np.zeros(image_size, dtype=np.float64)
    Y,X = np.meshgrid(range(image_size[1]), range(image_size[0]))
    R = np.sqrt((X-image_size[0]/2-ring_offset[1])**2 + (Y-image_size[1]/2-ring_offset[0])**2)
    img0[R > ring_start] = ring_strength
    img0[R > ring_end] = 0
    print(img.shape)
    print(img0.shape)
    img = img + img0
    return img

def GetTiltImage(img, data) :
    image_size = data['image_size']
    tilt_angle = data['tilt_angle']
    tilt_strength = data['tilt_strength']
    
    Y,X = np.meshgrid(range(image_size[1]), range(image_size[0]))
    X_ = X * np.cos(tilt_angle * np.pi / 180) + Y * np.sin(tilt_angle * np.pi / 180)
    Y_ = -X * np.sin(tilt_angle * np.pi / 180) + Y * np.cos(tilt_angle * np.pi / 180)
    img = img + X_*tilt_strength
    return img

def GetPolynomialImage(img, data) :
    image_size = data['image_size']
    tilt_angle = data['tilt_angle']
    polynomial_mode = data['polynomial_mode']
    polynomial_range = data['polynomial_range']
    polynomial_order = data['polynomial_order']
    polynomial_offset = data['polynomial_offset']

    Y,X = np.meshgrid(range(image_size[1]), range(image_size[0]))
    X = X - image_size[0]/2
    Y = Y - image_size[1]/2
    X_ = X * np.cos(tilt_angle * np.pi / 180) + Y * np.sin(tilt_angle * np.pi / 180) - polynomial_offset[1] 
    Y_ = -X * np.sin(tilt_angle * np.pi / 180) + Y * np.cos(tilt_angle * np.pi / 180) - polynomial_offset[0] 
    if(polynomial_mode == 1) :
        img0 = np.abs(Y_/polynomial_range)**polynomial_order * polynomial_range / np.abs(polynomial_range)
    elif(polynomial_mode == 2) :
        img0 = np.abs(X_/polynomial_range)**polynomial_order * polynomial_range / np.abs(polynomial_range)
    elif(polynomial_mode == 3) :
        R = np.abs(np.sqrt(X_**2 + Y_**2)/polynomial_range)
        img0 = R**polynomial_order * polynomial_range / np.abs(polynomial_range)
    img = img + img0
    return img

    
def GetCircleFillImage(img, data) :
    image_size = data['image_size']
    circle_offset = data['circle_offset']
    FLAG_CIRCLE_INTERIOR = data['FLAG_CIRCLE_INTERIOR']
    circle_radius = data['circle_radius']
    circle_setvalue = data['circle_setvalue']
    Y,X = np.meshgrid(range(image_size[1]), range(image_size[0]))
    R = np.sqrt((X-image_size[0]/2-circle_offset[1])**2 + (Y-image_size[1]/2-circle_offset[0])**2)
    if FLAG_CIRCLE_INTERIOR :
        img[R < circle_radius] = circle_setvalue
    else : 
        img[R > circle_radius] = circle_setvalue
    return img

def ApplyThreshold(img, data) :
    th_lower = data['th_lower']
    th_upper = data['th_upper']
    th_lowerset = data['th_lowerset']
    th_upperset = data['th_upperset']
    img[img < th_lower] = th_lowerset
    img[img > th_upper] = th_upperset
    return img

def AddOffsetValue(img, data) :
    addoffset_value = data['addoffset_value']
    img = img + addoffset_value
    return img

def DoRepSave(img, data) :
    for kk in range(10) :
        img = GetImageFromCode(img,data)
        Multi = data['Multi']
        savefilename = data['savefilename']
        head, tail = os.path.split(savefilename)
        basename,ext = os.path.splitext(tail)
        savefilename = head + "\\" + basename  + str(kk) + ext
        
        imgnew = np.copy(img) * Multi
        imgnew[imgnew < 0] = 0
        imgnew[imgnew > 1] = 1
        res = cv2.imwrite(savefilename, imgnew*255 )
        print('file save' , savefilename, ' return = ', res)
    return img

@app.callback(
    Output('figure-area', 'figure'),
    Output('store-img', 'data'),
    Input('button-new-image', 'n_clicks'),
    Input('button-code-draw', 'n_clicks'),
    Input('button-code-draw-rep', 'n_clicks'),
    Input('button-ring-draw', 'n_clicks'),
    Input('button-tilt-draw', 'n_clicks'),
    Input('button-circle-draw', 'n_clicks'),
    Input('button-threshold-apply', 'n_clicks'),
    Input('button-addoffset-draw', 'n_clicks'),
    Input('button-polynomial-draw', 'n_clicks'),
    State('store-data', 'data')    ,
    State('store-img', 'data')    
)
def ProcessButtonAction(n1,n2,n3,n4,n5,n6,n7,n8,n9, data, imgdata) :
    image_size = data['image_size']
    img = np.array(imgdata['img'])
    if(len(img) != image_size[0] * image_size[1]) : 
        img = np.zeros(image_size, dtype=np.float64)
    else :
        img =np.reshape(img, image_size)    
    #global img
    ctx = dash.callback_context
    if not ctx.triggered : 
        trigger_id = 'No trigger yet'
    else :
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if(img.shape[0] < 2) :
        img = np.zeros(image_size, dtype=np.float64)
    #print(trigger_id)
    
    #elif(trigger_id == 'task-component' and len(value) == 1) :
    #    Lats, photoncounts, xopts = CheckAndPhaseOutput(sidebar.currDirectory, prevfilename, sidebar.AnalysisMode)
    if(trigger_id == 'button-new-image  ') :
        img = GetNewImage(img,data)
    elif(trigger_id == 'button-code-draw') :
        img = GetImageFromCode(img,data)
    elif(trigger_id == 'button-code-draw-rep') :
        print("+++++++++++BUTTON+++++++++++++")
        img = DoRepSave(img, data)

    elif(trigger_id == 'button-ring-draw') :
        img = GetRingImage(img,data)
    elif(trigger_id == 'button-tilt-draw') :
        img = GetTiltImage(img,data)
    elif(trigger_id == 'button-polynomial-draw') :
        img = GetPolynomialImage(img,data)
    elif(trigger_id == 'button-circle-draw') :
        img = GetCircleFillImage(img,data)
    elif(trigger_id == 'button-threshold-apply') :
        img = ApplyThreshold(img,data)
    elif(trigger_id == 'button-addoffset-draw') :
        img = AddOffsetValue(img,data)
    else :
        img = img
    fig = px.imshow(img)
    imgdata['img'] = list(img.reshape(-1))
    return fig, imgdata


############# TEMPLATE CODES ###########

def lineXCode() :
    res = "for i in range(J.shape[0]) : \n    for j in range(J.shape[1]) : \n"
    res += "        if(j==midpt_j) : \n"
    res += "            J[i,j] = 1 \n"
    return res

def lineYCode() :
    res = "for i in range(J.shape[0]) : \n    for j in range(J.shape[1]) : \n"
    res += "        if(i==midpt_i) : \n"
    res += "            J[i,j] = 1 \n"
    return res

def StripeXCode() :
    res = "for i in range(J.shape[0]) : \n    for j in range(J.shape[1]) : \n"
    res += "        if(np.mod(j,2)==1) : \n"
    res += "            J[i,j] = 1 \n"
    return res

def StripeYCode() :
    res = "for i in range(J.shape[0]) : \n    for j in range(J.shape[1]) : \n"
    res += "        if(np.mod(i,2)==1) : \n"
    res += "            J[i,j] = 1 \n"
    return res


@app.callback(
    Output('text-area-demo', 'value'),
    Input('button-lineX-code', 'n_clicks'),
    Input('button-lineY-code', 'n_clicks'),
    Input('button-stripeX-code', 'n_clicks'),
    Input('button-stripeY-code', 'n_clicks'),
)
def WriteCode(n1,n2,n3,n4) :
    global img
    ctx = dash.callback_context
    if not ctx.triggered : 
        trigger_id = 'No trigger yet'
    else :
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if(trigger_id == 'button-lineX-code') :
        return lineXCode()
    elif(trigger_id == 'button-lineY-code') :
        return lineYCode()
    elif(trigger_id == 'button-stripeX-code') :
        return StripeXCode()
    elif(trigger_id == 'button-stripeY-code') :
        return StripeYCode()
    return "for i in range(J.shape[0]) : \n    for j in range(J.shape[1]) : \n        J[i,j]=np.mod((i+j),2)"

########## CALLBACK SET PARAMETERS ##########
@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-Multi','valid'),
    Input('input-Multi','value'),
    State("store-data", 'data'),
)
def SetW(value, data) :
    #global Multi
    try :
        Multi = float(value)
        print(data)
        data['Multi'] = Multi
    except :
        Multi = 0
        data['Multi'] = 0
        return data, False
    return data, True


@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-W','valid'),
    Input('input-W','value'),
    State("store-data", 'data'),
)
def SetW(value, data) :
    #global W 
    try :
        W = float(value)
        data['W'] = W
    except :
        W = 0
        data['W'] = W
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-D','valid'),
    Input('input-D','value'),
    State("store-data", 'data'),
)
def SetW(value,data) :
    #global D 
    try :
        D = float(value)
        data['D'] = D
    except :
        D = 0
        data['D'] = D
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-Neighbor-TH','valid'),
    Input('input-Neighbor-TH','value'),
    State("store-data", 'data'),
)
def SetNeighborTH(value,data) :
    # global Neibhor_TH
    try :
        temp = float(value)
        if temp <0 and temp > 1 :
            return False
        Neighbor_TH = temp
        data['Neighbor_TH'] = Neighbor_TH
    except :
        Neighbor_TH = 0
        data['Neighbor_TH'] = Neighbor_TH
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-atomplane-size','valid'),
    Input('input-atomplane-size','value'),
    State("store-data", 'data'),
)
def setAtomPlaneSize(value, data) :
    #global AtomPlane_size
    try :
        temp = value.split(',')
        AtomPlane_size = [int(temp[0]), int(temp[1])]
        data['AtomPlane_size'] = AtomPlane_size
    except :
        AtomPlane_size = [0,0]
        data['AtomPlane_size'] = AtomPlane_size
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-image-size','valid'),
    Input('input-image-size','value'),
    State("store-data", 'data'),
)
def setAtomPlaneSize(value, data) :
    #global image_size
    try :
        temp = value.split(',')
        image_size = [int(temp[0]), int(temp[1])]
        data['image_size'] = image_size
    except :
        image_size = [0,0]
        data['image_size'] = image_size
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('div-empty-switch','children'),
    Input('switch-stick-neighbor','value'),
    State("store-data", 'data'),
)
def setAtomPlaneSize(value, data) :
    #global FLAG_STICK_NEIGHBOR
    try :
        FLAG_STICK_NEIGHBOR = value
        data['FLAG_STICK_NEIGHBOR'] = FLAG_STICK_NEIGHBOR
    except :
        FLAG_STICK_NEIGHBOR = False
        data['FLAG_STICK_NEIGHBOR'] = FLAG_STICK_NEIGHBOR
        return data, ""
    return data, ""


########## CALLBACK TRANSFORM MATRIX ##########
@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-ang1','valid'),
    Input('input-ang1','value'),
    State("store-data", 'data'),
)
def SetAng1(value, data) :
    #global ang1
    try :
        temp = float(value)
        ang1 = temp
        data['ang1'] = ang1
    except :
        ang1 = 0
        data['ang1'] = ang1
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-ang2','valid'),
    Input('input-ang2','value'),
    State("store-data", 'data'),
)
def SetAng2(value, data) :
    #global ang2
    try :
        temp = float(value)
        ang2 = temp
        data['ang2'] = ang2
    except :
        ang2 = 0
        data['ang2'] = ang2
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-L1','valid'),
    Input('input-L1','value'),
    State("store-data", 'data'),
)
def SetL1(value, data) :
    #global L1
    try :
        temp = float(value)
        L1 = temp
        data['L1'] = L1
    except :
        L1 = 0
        data['L1'] = L1
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-L2','valid'),
    Input('input-L2','value'),
    State("store-data", 'data'),
)
def SetL2(value, data) :
    #global L2
    try :
        temp = float(value)
        L2 = temp
        data['L2'] = L2
    except :
        L2 = 0
        data['L2'] = L2
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-offset','valid'),
    Input('input-offset','value'),
    State("store-data", 'data'),
)
def setAtomPlaneSize(value, data) :
    #global offset
    try :
        temp = value.split(',')
        offset = [int(temp[0]), int(temp[1])]
        data['offset'] = offset
    except :
        offset = [0,0]
        data['offset'] = offset
        return data, False
    return data, True

########### CALLBACK ACCORDION ###########
########### RING PATTERN #################
@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-ring-start','valid'),
    Input('input-ring-start','value'),
    State("store-data", 'data'),
)
def SetRingStart(value, data) :
    #global ring_start
    try :
        temp = float(value)
        ring_start = temp
        data['ring_start'] = ring_start
    except :
        ring_start = 0
        data['ring_start'] = ring_start
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-ring-end','valid'),
    Input('input-ring-end','value'),
    State("store-data", 'data'),
)
def SetRingEnd(value, data) :
    #global ring_end
    try :
        temp = float(value)
        ring_end = temp
        data['ring_end'] = ring_end
    except :
        ring_end = 0
        data['ring_end'] = ring_end
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-ring-strength','valid'),
    Input('input-ring-strength','value'),
    State("store-data", 'data'),
)
def SetRingStrength(value, data) :
    #global ring_strength
    try :
        temp = float(value)
        ring_strength = temp
        data['ring_strength'] = ring_strength
    except :
        ring_strength = 0
        data['ring_strength'] = ring_strength
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-ring-offset','valid'),
    Input('input-ring-offset','value'),
    State("store-data", 'data'),
)
def SetRingOffset(value, data) :
    #global ring_offset
    try :
        temp = value.split(',')
        ring_offset = [int(temp[0]), int(temp[1])]
        data['ring_offset'] = ring_offset
    except :
        ring_offset = [0,0]
        data['ring_offset'] = ring_offset
        return data, False
    return data, True

########### TILT PATTERN #################
@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-tilt-angle','valid'),
    Input('input-tilt-angle','value'),
    State("store-data", 'data'),
)
def SetTiltAngle(value, data) :
    #global tilt_angle
    try :
        temp = float(value)
        tilt_angle = temp
        data['tilt_angle'] = tilt_angle
    except :
        tilt_angle = 0
        data['tilt_angle'] = tilt_angle
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-tilt-strength','valid'),
    Input('input-tilt-strength','value'),
    State("store-data", 'data'),
)
def SetTiltStrength(value, data) :
    # global tilt_strength
    try :
        temp = float(value)
        tilt_strength = temp
        data['tilt_strength'] = tilt_strength
    except :
        tilt_strength = 0
        data['tilt_strength'] = tilt_strength
        return data, False
    return data, True
########### ADD OFFSET #################
@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-addoffset-value','valid'),
    Input('input-addoffset-value','value'),
    State("store-data", 'data'),
)
def SetPolynomialOrder(value, data) :
    #global addoffset_value
    try :
        temp = float(value)
        addoffset_value = temp
        data['addoffset_value'] = addoffset_value
    except :
        addoffset_value = 0
        data['addoffset_value'] = addoffset_value
        return data, False
    return data, True

########### POLYNOMIAL POTENTIAL #################
@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-polynomial-order','valid'),
    Input('input-polynomial-order','value'),
    State("store-data", 'data'),
)
def SetPolynomialOrder(value, data) :
    #global polynomial_order
    try :
        temp = float(value)
        polynomial_order = temp
        data['polynomial_order'] = polynomial_order
    except :
        polynomial_order = 0
        data['polynomial_order'] = polynomial_order
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-polynomial-range','valid'),
    Input('input-polynomial-range','value'),
    State("store-data", 'data'),
)
def SetPolynomialRange(value, data) :
    #global polynomial_range
    try :
        temp = float(value)
        polynomial_range = temp
        data['polynomial_range'] = polynomial_range
    except :
        polynomial_range = 0
        data['polynomial_range'] = polynomial_range
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-polynomial-offset','valid'),
    Input('input-polynomial-offset','value'),
    State("store-data", 'data'),
)
def SetCircleOffset(value, data) :
    #global polynomial_offset
    try :
        temp = value.split(',')
        polynomial_offset = [int(temp[0]), int(temp[1])]
        data['polynomial_offset'] = polynomial_offset
    except :
        polynomial_offset = [0,0]
        data['polynomial_offset'] = polynomial_offset
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Input('radioitems-polynomial','value'),
    State("store-data", 'data'),
)
def SetPolynomialRadioItem(value, data) :
    #global polynomial_mode
    polynomial_mode = value
    data['polynomial_mode'] = polynomial_mode
    return data


########### FILL PATTERN #################
@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-circle-radius','valid'),
    Input('input-circle-radius','value'),
    State("store-data", 'data'),
)
def SetCircleRadius(value, data) :
    #global circle_radius
    try :
        temp = float(value)
        circle_radius = temp
        data['circle_radius'] = circle_radius
    except :
        circle_radius = 0
        data['circle_radius'] = circle_radius
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-circle-setvalue','valid'),
    Input('input-circle-setvalue','value'),
    State("store-data", 'data'),
)
def SetCircleSetValue(value, data) :
    #global circle_setvalue
    try :
        temp = float(value)
        circle_setvalue = temp
        data['circle_setvalue'] = circle_setvalue
    except :
        circle_setvalue = 0
        data['circle_setvalue'] = circle_setvalue
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Input('radioitems-int-ext','value'),
    State("store-data", 'data'),
)
def SetCircleRadioitem(value, data) :
    #global FLAG_CIRCLE_INTERIOR
    FLAG_CIRCLE_INTERIOR = (value==1)
    data['FLAG_CIRCLE_INTERIOR'] = FLAG_CIRCLE_INTERIOR
    return data

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-circle-offset','valid'),
    Input('input-circle-offset','value'),
    State("store-data", 'data'),
)
def SetCircleOffset(value, data) :
    #global circle_offset
    try :
        temp = value.split(',')
        circle_offset = [int(temp[0]), int(temp[1])]
        data['circle_offset'] = circle_offset
    except :
        circle_offset = [0,0]
        data['circle_offset'] = circle_offset
        return data, False
    return data, True

########### THRESHOLD SET #################
@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-threshold-lower','valid'),
    Input('input-threshold-lower','value'),
    State("store-data", 'data'),
)
def SetThresholdLower(value, data) :
    #global th_lower
    try :
        temp = float(value)
        th_lower = temp
        data['th_lower'] = th_lower
    except :
        th_lower = 0
        data['th_lower'] = th_lower
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-threshold-lowerset','valid'),
    Input('input-threshold-lowerset','value'),
    State("store-data", 'data'),
)
def SetThresholdLowerSet(value, data) :
    #global th_lowerset
    try :
        temp = float(value)
        th_lowerset = temp
        data['th_lowerset'] = th_lowerset
    except :
        th_lowerset = 0
        data['th_lowerset'] = th_lowerset
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-threshold-upper','valid'),
    Input('input-threshold-upper','value'),
    State("store-data", 'data'),
)
def SetThresholdUpper(value, data) :
    #global th_upper
    try :
        temp = float(value)
        th_upper = temp
        data['th_upper'] = th_upper
    except :
        th_upper = 0
        data['th_upper'] = th_upper
        return data, False
    return data, True

@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-threshold-upperset','valid'),
    Input('input-threshold-upperset','value'),
    State("store-data", 'data'),
)
def SetThresholdUpperSet(value, data) :
    #global th_upperset
    try :
        temp = float(value)
        th_upperset = temp
        data['th_upperset'] = th_upperset
    except :
        th_upperset = 0
        data['th_upperset'] = th_upperset
        return data, False
    return data, True

########### SAVE FILE #################
@app.callback(
    Output("store-data", 'data',allow_duplicate=True),
    Output('input-path-filename','valid'),
    Input('input-path-filename','value'),
    State("store-data", 'data'),
)
def SetSaveFileName(value, data) :
    #global savefilename
    try :
        savefilename = value
        data['savefilename'] = savefilename
    except :
        savefilename = ""
        data['savefilename'] = savefilename
        return data, False
    return data, True

@app.callback(
    Output('div-path-empty','children'),
    Input('button-path-save','n_clicks'),
    State("store-data", 'data'),
    State("store-img", 'data'),
)
def SaveFile(n_clicks, data, imgdata) :
    image_size = data['image_size']
    img = np.array(imgdata['img'])
    if(len(img) != image_size[0] * image_size[1]) : 
        img = np.zeros(image_size, dtype=np.float64)
    else :
        img =np.reshape(img, image_size)    
    
    Multi = data['Multi']
    savefilename = data['savefilename']
    imgnew = np.copy(img) * Multi
    imgnew[imgnew < 0] = 0
    imgnew[imgnew > 1] = 1
    res = cv2.imwrite(savefilename, imgnew*255 )
    return "File save : ", savefilename, "\n Response: " + str(res)
#imgtool.Example()

if __name__ == "__main__":
    #app.run_server(debug=True)
    app.run_server(debug=True, port=8088, host='0.0.0.0')