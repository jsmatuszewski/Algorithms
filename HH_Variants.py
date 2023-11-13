import dash
from dash import dcc
from dash import Dash, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy
import networkx as nx
import random

def getNRounds(stuff, rounds):

    edges = stuff[0]
    complete = stuff[2]

    edgestoReturn = [[edge[1],edge[2]] for edge in edges if edge[0]<=rounds]

    return edgestoReturn, stuff[1], complete

def handle(inputs):

    inputs = inputs.split(',')
    inputs = [int(x) for x in inputs]

    return inputs

def MaxHH(SEQ):

    labels = [i for i in range(1,len(SEQ)+1)]
    edges = []
    results = {}
    maxDegree = SEQ[0]

    for j in range(0,maxDegree+1):

        results[j] = {'counts' : 0, 'vertices':[]}

    for i in range(0,len(SEQ)):

        degree = SEQ[i]
        label = labels[i]

        results[degree]['counts']+=1
        results[degree]['vertices'].append(label)

    rounds = 1
    while results[0]['counts'] < len(SEQ):

        # Check if max degree bin count is 0, if it is go to next bin...
        if results[maxDegree]['counts'] == 0:

            maxDegree -= 1

        #if it isnt, take your pivot
        else:

            #Take pivot and remove from bin, decrease count by 1.
            results[maxDegree]['vertices'].sort()
            pivot = results[maxDegree]['vertices'][0]
            results[maxDegree]['vertices']= results[maxDegree]['vertices'][1:]
            results[maxDegree]['counts'] = results[maxDegree]['counts']-1

            #Add to done bin
            results[0]['vertices'].append(pivot)
            results[0]['counts']+=1


            #Connection not yet added
            added = 0
            degreesLeft = maxDegree
            bin_ = maxDegree

            #while connection not added
            i=0
            LR = []
            while degreesLeft > 0:

                #In subsequent bins there are a few cases:

                #Case 1... Bin is empty (counts = 0)
                #Case 2... Bin contains only vertices already added (counts = numAdded)
                #Case 3....Bin contains new vertices and vertices already added (counts > numAdded > 0)
                #Case 4... Bin only contains vertices not yet added (counts > 0 = numAdded)

                #Case 1...
                if results[bin_]['counts'] == 0:

                    LR = []

                #Case 2...
                else:

                    #We want to add everything after what came from the last round,
                    #within however many vertices we have left to add...

                    #Take all vertices in next bin and sort them...
                    inBin = results[bin_]['vertices']
                    inBin.sort()
                    #Take all vertices in bin that weren't in the last round....
                    toAdd = [x for x in inBin if x not in LR][:degreesLeft]

                    #Add an edge for each of these vertices
                    edges= edges + [[rounds,pivot,x] for x in toAdd]

                    #Now the bin should only contain things from last round, and things we couldn't grab this round
                    results[bin_]['vertices'] = list(set(inBin)-set(toAdd))
                    results[bin_]['counts'] -= len(toAdd)

                    results[bin_-1]['vertices'] = toAdd + results[bin_-1]['vertices']
                    results[bin_-1]['counts'] += len(toAdd)



                    LR = toAdd

                degreesLeft -= len(LR)
                bin_-=1

            rounds+=1

    steps = [x[0] for x in edges]
    complete = max(steps)
    return edges,len(SEQ),complete

##With alphabetic sorting
def MinHH(SEQ):

    #Just initialize labels...
    labels = [i for i in range(1,len(SEQ)+1)]
    edges = []
    results = {}
    maxDegree = SEQ[0]

    for j in range(0,maxDegree+1):

        results[j] = {'counts' : 0, 'vertices':[]}

    for i in range(0,len(SEQ)):

        degree = SEQ[i]
        label = labels[i]

        results[degree]['counts']+=1
        results[degree]['vertices'].append(label)


    minDegree = SEQ[-1]
    degreesLeft = 0

    rounds = 1
    while results[0]['counts'] < len(SEQ):

        # Check if max degree bin count is 0, if it is go to next bin...
        if (results[minDegree]['counts'] == 0):

            minDegree += 1

        #if it isnt, take your pivot
        else:

            #Take pivot and remove from bin, decrease count by 1.
            results[minDegree]['vertices'].sort()
            pivot = results[minDegree]['vertices'][0]
            results[minDegree]['vertices']= results[minDegree]['vertices'][1:]
            results[minDegree]['counts'] = results[minDegree]['counts']-1

            #Add to done bin
            results[0]['vertices'].append(pivot)
            results[0]['counts']+=1

            #Connection not yet added
            added = 0
            degreesLeft = minDegree
            bin_ = maxDegree

            #while connection not added
            i=0
            LR = []

            while degreesLeft > 0:

                #In subsequent bins there are a few cases:

                #Case 1... Bin is empty (counts = 0)
                #Case 2... Bin contains only vertices already added (counts = numAdded)
                #Case 3....Bin contains new vertices and vertices already added (counts > numAdded > 0)
                #Case 4... Bin only contains vertices not yet added (counts > 0 = numAdded)

                #Case 1...
                if results[bin_]['counts'] == 0:

                    LR = []

                #Case 2...
                else:

                    #We want to add everything after what came from the last round,
                    #within however many vertices we have left to add...

                    #Take all vertices in next bin and sort them...
                    inBin = results[bin_]['vertices']
                    inBin.sort()
                    #Take all vertices in bin that weren't in the last round....
                    toAdd = [x for x in inBin if x not in LR][:degreesLeft]

                    #Add an edge for each of these vertices
                    edges= edges + [[rounds,pivot,x] for x in toAdd]

                    #Now the bin should only contain things from last round, and things we couldn't grab this round
                    results[bin_]['vertices'] = list(set(inBin)-set(toAdd))
                    results[bin_]['counts'] -= len(toAdd)

                    results[bin_-1]['vertices'] = toAdd + results[bin_-1]['vertices']
                    results[bin_-1]['counts'] += len(toAdd)

                    if minDegree > (bin_-1):

                        minDegree = bin_-1

                    LR = toAdd

                degreesLeft -= len(LR)
                bin_-=1
        rounds+=1

    steps = [x[0] for x in edges]
    complete = max(steps)
    return edges, len(SEQ),complete

def UR_HH(SEQ):

    ops = 0
    labels = [i for i in range(1,len(SEQ)+1)]
    edges = []
    results = {}
    backLook = {}
    maxDegree = SEQ[0]

    for j in range(0,maxDegree+1):

        results[j] = {'counts' : 0, 'vertices':[]}

    for i in range(0,len(SEQ)):

        degree = SEQ[i]
        label = labels[i]

        backLook[label] = {'bin':degree}

        results[degree]['counts']+=1
        results[degree]['vertices'].append(label)

    rounds = 1
    while len(labels) > 0:

        #Grab a random pivot
        pivot = random.choice(labels)

        #Move this pivot to 0 bin
        bin_ = backLook[pivot]['bin']
        labels.remove(pivot)
        del backLook[pivot]

        #Remove this pivot from it's bin in results and decrease count....
        results[bin_]['vertices'].remove(pivot)
        results[bin_]['counts']-=1

        #Add to done bin...
        results[0]['vertices'].append(pivot)
        results[0]['counts']+=1

        #connection not yet added
        added = 0
        degreesLeft = bin_ #Degrees left to add is equal to bin it was in...

        #while connection not added
        i = 0
        LR = []

        while degreesLeft > 0:

            #In subsequent bins there are a few cases:

            #Case 1... Bin is empty (counts = 0)
            #Case 2... Bin contains only vertices already added (counts = numAdded)
            #Case 3....Bin contains new vertices and vertices already added (counts > numAdded > 0)
            #Case 4... Bin only contains vertices not yet added (counts > 0 = numAdded)

            #Case 1... check max degree bin. if nothing is inside, keep moving.
            if results[maxDegree]['counts'] == 0:

                #decrement max degree
                maxDegree -= 1

                toAdd = []
                LR = []

            #Case 2... Bin isn't empty
            else:

                #Take all vertices in next bin and sort them...
                inBin = results[maxDegree]['vertices']
                inBin.sort()
                #Take all vertices in bin that weren't in the last round....
                toAdd = [x for x in inBin if x not in LR][:degreesLeft]

                #Add an edge for each of these vertices
                edges= edges + [[rounds,pivot,x] for x in toAdd]

                #Now the bin should only contain things from last round, and things we couldn't grab this round
                results[maxDegree]['vertices'] = list(set(inBin)-set(toAdd))
                results[maxDegree]['counts'] -= len(toAdd)

                results[maxDegree-1]['vertices'] = toAdd + results[maxDegree-1]['vertices']
                results[maxDegree-1]['counts'] += len(toAdd)

                for added in toAdd:

                    binH = backLook[added]['bin']

                    if binH == 1:

                        labels.remove(added)
                        backLook.pop(added, None)

                    else:
                        backLook[added]['bin'] = binH-1


                LR = toAdd

            degreesLeft -=len(toAdd)
            bin_-=1
        rounds+=1

    steps = [x[0] for x in edges]
    complete = max(steps)
    return edges, len(SEQ),complete

def number_to_letter(num):
    if 1 <= num <= 26:
        return chr(ord('A') + num - 1)
    else:
        return str(num)

def PR_HH(SEQ):

    ops = 0
    labels = [i for i in range(1,len(SEQ)+1)]
    edges = []
    results = {}
    backLook = {}
    maxDegree = SEQ[0]

    for j in range(0,maxDegree+1):

        results[j] = {'counts' : 0, 'vertices':[]}

    for i in range(0,len(SEQ)):

        degree = SEQ[i]
        label = labels[i]

        backLook[label] = degree

        results[degree]['counts']+=1
        results[degree]['vertices'].append(label)

    labels = [key for key, value in backLook.items() for _ in range(value)]

    rounds = 1
    while len(labels) > 0:

        #Grab a random pivot
        pivot = random.choice(labels)

        #Move this pivot to 0 bin
        bin_ = backLook[pivot]
        labels.remove(pivot)
        del backLook[pivot]

        #Remove this pivot from it's bin in results and decrease count....
        results[bin_]['vertices'].remove(pivot)
        results[bin_]['counts']-=1

        #Add to done bin...
        results[0]['vertices'].append(pivot)
        results[0]['counts']+=1

        #connection not yet added
        added = 0
        degreesLeft = bin_ #Degrees left to add is equal to bin it was in...

        #while connection not added
        i = 0
        LR = []

        while degreesLeft > 0:

            labels = [key for key, value in backLook.items() for _ in range(value)]
            #In subsequent bins there are a few cases:

            #Case 1... Bin is empty (counts = 0)
            #Case 2... Bin contains only vertices already added (counts = numAdded)
            #Case 3....Bin contains new vertices and vertices already added (counts > numAdded > 0)
            #Case 4... Bin only contains vertices not yet added (counts > 0 = numAdded)

            #Case 1... check max degree bin. if nothing is inside, keep moving.
            if results[maxDegree]['counts'] == 0:

                #decrement max degree
                maxDegree -= 1

                toAdd = []
                LR = []

            #Case 2... Bin isn't empty
            else:

                #Take all vertices in next bin and sort them...
                inBin = results[maxDegree]['vertices']
                inBin.sort()
                #Take all vertices in bin that weren't in the last round....
                toAdd = [x for x in inBin if x not in LR][:degreesLeft]

                #Add an edge for each of these vertices
                edges = edges + [[rounds,pivot,x] for x in toAdd]

                #Now the bin should only contain things from last round, and things we couldn't grab this round
                results[maxDegree]['vertices'] = list(set(inBin)-set(toAdd))
                results[maxDegree]['counts'] -= len(toAdd)

                results[maxDegree-1]['vertices'] = toAdd + results[maxDegree-1]['vertices']
                results[maxDegree-1]['counts'] += len(toAdd)

                for added in toAdd:

                    binH = backLook[added]

                    if binH == 1:

                        labels.remove(added)
                        backLook.pop(added, None)

                    else:
                        backLook[added] = binH-1

                LR = toAdd


            degreesLeft -=len(toAdd)
            bin_-=1
        rounds+=1
    steps = [x[0] for x in edges]
    complete = max(steps)
    return edges, len(SEQ), complete

def Par_HH(SEQ,par):

    ops = 0
    labels = [i for i in range(1,len(SEQ)+1)]
    edges = []
    results = {}
    backLook = {}
    maxDegree = SEQ[0]

    for j in range(0,maxDegree+1):

        results[j] = {'counts' : 0, 'vertices':[]}

    for i in range(0,len(SEQ)):

        degree = SEQ[i]
        label = labels[i]

        backLook[label] = degree

        results[degree]['counts']+=1
        results[degree]['vertices'].append(label)

    nums = [value**par for key,value in backLook.items()]
    denom = sum(nums)+(0.00000000000000000000000000000001)
    probs = [x/denom for x in nums]

    rounds = 1
    while len(labels) > 0:

        nums = [value**par for key,value in backLook.items()]
        denom = sum(nums)+(0.00000000000000000000000000000001)
        probs = [x/denom for x in nums]

        #Grab a random pivot
        pivot = random.choices(labels, probs, k=1)[0]

        #Move this pivot to 0 bin
        bin_ = backLook[pivot]
        labels.remove(pivot)
        del backLook[pivot]

        #Remove this pivot from it's bin in results and decrease count....
        results[bin_]['vertices'].remove(pivot)
        results[bin_]['counts']-=1

        #Add to done bin...
        results[0]['vertices'].append(pivot)
        results[0]['counts']+=1

        #connection not yet added
        added = 0
        degreesLeft = bin_ #Degrees left to add is equal to bin it was in...

        #while connection not added
        i = 0
        LR = []

        while degreesLeft > 0:

            #In subsequent bins there are a few cases:

            #Case 1... Bin is empty (counts = 0)
            #Case 2... Bin contains only vertices already added (counts = numAdded)
            #Case 3....Bin contains new vertices and vertices already added (counts > numAdded > 0)
            #Case 4... Bin only contains vertices not yet added (counts > 0 = numAdded)

            #Case 1... check max degree bin. if nothing is inside, keep moving.
            if results[maxDegree]['counts'] == 0:

                #decrement max degree
                maxDegree -= 1

                toAdd = []
                LR = []

            #Case 2... Bin isn't empty
            else:

                #Take all vertices in next bin and sort them...
                inBin = results[maxDegree]['vertices']
                inBin.sort()
                #Take all vertices in bin that weren't in the last round....
                toAdd = [x for x in inBin if x not in LR][:degreesLeft]

                #Add an edge for each of these vertices
                edges = edges + [[rounds,pivot,x] for x in toAdd]

                #Now the bin should only contain things from last round, and things we couldn't grab this round
                results[maxDegree]['vertices'] = list(set(inBin)-set(toAdd))
                results[maxDegree]['counts'] -= len(toAdd)

                results[maxDegree-1]['vertices'] = toAdd + results[maxDegree-1]['vertices']
                results[maxDegree-1]['counts'] += len(toAdd)

                for added in toAdd:

                    binH = backLook[added]

                    if binH == 1:

                        labels.remove(added)
                        backLook.pop(added, None)

                    else:
                        backLook[added] = binH-1

                LR = toAdd

            degreesLeft -=len(toAdd)
            bin_-=1
        rounds+=1
    steps = [x[0] for x in edges]
    complete = max(steps)
    return edges,len(SEQ),complete

layout = go.Layout(
    autosize=False,
    height = 800,
    width = 800,

)
IETsr = 2
global prev_sequence, min_hh, max_hh, ur_hh, pr_hh, par_hh, minhh, maxhh, urhh, prhh, parhh, param
prev_sequence ='2,2,2'
min_hh = MinHH(handle(prev_sequence))
minhh = min_hh
max_hh = MaxHH(handle(prev_sequence))
maxhh = max_hh
ur_hh = UR_HH(handle(prev_sequence))
urhh = ur_hh
pr_hh = PR_HH(handle(prev_sequence))
prhh = pr_hh
par_hh = Par_HH(handle(prev_sequence),-100)
parhh = par_hh
param = 0

def progressBar(progress, finishes):

    x = [progress/x for x in finishes]
    colors = ['blue', 'green', 'orange', 'red', 'purple']

    fig = go.Figure(go.Bar(
        x=x,
        y=['Min HH', 'Max HH', 'UR HH', 'PR HH', 'Par HH'],
        orientation='h',
        marker=dict(color=colors)))

    fig.update_xaxes(
        range=[0, 1],
        showgrid=False,
        showticklabels=False,  # Set to False to remove x-axis ticks
        mirror=True
    )


    fig.update_layout(
        title=dict(
            text='Progress Bars',
            font=dict(size=20, color='black'),
            y=0.92,  # Adjust this value to center the title vertically
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin = dict(t=50, b=10, l=10, r=10, pad=0),
        xaxis=dict(linecolor='black', showgrid=False, showticklabels=False, mirror=True),
        yaxis=dict(linecolor='black', showgrid=False, showticklabels=True, mirror=True),

    )

    return fig


def networkGraph(edges,length,title,color):

    edges = edges
    G = nx.Graph()
    G.add_nodes_from([number_to_letter(i) for i in range(1,length+1)])
    edges = [[number_to_letter(edge[0]),number_to_letter(edge[1])] for edge in edges]
    G.add_edges_from(edges)
    pos = nx.circular_layout(G)

    # edges trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color='black', width=1),
        hoverinfo='none',
        showlegend=False,
        mode='lines')

    # nodes trace
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, text=text,
        mode='markers+text',
        showlegend=False,
        hoverinfo='none',
        marker=dict(
            color='pink',
            size=50,
            line=dict(color='black', width=1)))

    # layout
    layout = dict(plot_bgcolor='white',
                  paper_bgcolor='white',
                  margin=dict(t=50, b=10, l=10, r=10, pad=0),
                  xaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True),
                  yaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True,
                             title_font=dict(color='red')),
                  annotations=[
                      dict(
                          x=0.5,
                          y=1.10,
                          xref='paper',
                          yref='paper',
                          text=title,
                          showarrow=False,
                          font=dict(size=20, color= color)
                      )
                  ] )



    # figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return fig

# Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1('Havel Hakimi',style={'textAlign': 'center','color': 'black','font-size': 40}),
    html.I('Enter a degree sequence: ',style={'color': 'black','font-size': 25}),
    dcc.Input(id='sequence', type='text', value='2,2,2', debounce=True),
    html.Br(),
    html.I('Enter parameter value for ParHH: ',style={'color': 'black','font-size': 25}),
    dcc.Input(id='parameter', type='text', value='0', debounce=True),
    html.Br(),
    html.I('Advance rounds completed:',style={'color': 'black','font-size': 25}),
    dcc.Slider(
        id='3DHR Slider',
        min=0,
        max=IETsr,
        value=0,
        marks={i: {'label': f"{i}", 'style': {'font-size': '30px'}} for i in range(0, IETsr + 1)},
        step=1,
    ),
    html.Div(children = [
        dcc.Graph(id='my-graph1',style={'width': '40vh', 'height': '40vh','display': 'inline-block'}),
        dcc.Graph(id='my-graph2',style={'width': '40vh', 'height': '40vh','display': 'inline-block'}),
        dcc.Graph(id="graph", style={'width': '40vh', 'height': '40vh','display': 'inline-block'})
    ]),
    html.Div(children = [
        dcc.Graph(id='my-graph3',style={'width': '40vh', 'height': '40vh','display': 'inline-block'}),
        dcc.Graph(id='my-graph4',style={'width': '40vh', 'height': '40vh','display': 'inline-block'}),
        dcc.Graph(id='my-graph5',style={'width': '40vh', 'height': '40vh','display': 'inline-block'}),
    ])
],style={'margin': 'auto', 'width': '80%'})


@app.callback([Output('3DHR Slider', 'max'),Output('3DHR Slider', 'value'),Output('3DHR Slider', 'marks'),Output('my-graph2', 'figure'), Output('my-graph1', 'figure'),
               Output('my-graph3', 'figure'), Output('my-graph4', 'figure'),
               Output('my-graph5', 'figure'),Output('graph', 'figure')],
              [Input('3DHR Slider', 'value'),Input('sequence', 'value'),Input('parameter', 'value')])


def update_graph(steps,sequence,param):
    # Your callback logic here

    global prev_sequence, min_hh, max_hh, ur_hh, pr_hh, par_hh, minhh, maxhh, urhh, prhh, parhh

    if (sequence == prev_sequence):

        min_hh = getNRounds(minhh,steps)
        max_hh = getNRounds(maxhh,steps)
        ur_hh = getNRounds(urhh,steps)
        pr_hh= getNRounds(prhh,steps)
        par_hh= getNRounds(parhh,steps)
        value = steps

    else:

        prev_sequence = sequence
        minhh = MinHH(handle(sequence))
        maxhh = MaxHH(handle(sequence))
        urhh = UR_HH(handle(sequence))
        prhh = PR_HH(handle(sequence))
        parhh = Par_HH(handle(sequence),float(param))

        min_hh = getNRounds(minhh,0)
        max_hh = getNRounds(maxhh,0)
        ur_hh = getNRounds(urhh,0)
        pr_hh= getNRounds(prhh,0)
        par_hh= getNRounds(parhh,0)
        value = 0

    IETsr = max([min_hh[2],max_hh[2],ur_hh[2],pr_hh[2],par_hh[2]])
    marks={i: f"{i}" for i in range(0,IETsr+1)}

    # Example figure, replace this with your actual figure creation logic
    figure1 = networkGraph(min_hh[0],min_hh[1],'Min HH','blue')
    figure2 = networkGraph(max_hh[0],max_hh[1],'Max HH','green')
    figure3 = networkGraph(ur_hh[0],ur_hh[1],'UR HH','orange')
    figure4 = networkGraph(pr_hh[0],pr_hh[1],'PR HH','red')
    figure5 = networkGraph(par_hh[0],par_hh[1],'PAR HH','purple')
    figure6 = progressBar(value,[min_hh[2],max_hh[2],ur_hh[2],pr_hh[2],par_hh[2]])

    return IETsr, value, marks, figure1, figure2, figure3, figure4, figure5, figure6

app.run_server(host='0.0.0.0', port=10000)
