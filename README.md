<font size="6">
<p align="center"><b>pyLocalGLMnet</b> <br>
Eine Python Implementierung des Richman/Wüthrich-Ansatzes</p>

# Inhalt 
[__Abbildungsverzeichnis__](#abbildungsverzeichnis)<br>
[__Dependencies__](#dependencies)<br>
<br>
[__1. Einleitung__](#1-bullet) <br>
<br>
[__2. Datensatz 1: Künstlicher Datensatz__](#2-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.1 Künstlichen Datensatz erzeugen](#2_1-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.2 LocalGLMnet](#2_2-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.2.1 GLM](#2_2_1-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.2.2 LocalGLMnet](#2_2_2-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.2.3 Performance Benchmark](#2_2_3-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.3 Auswertung](#2_3-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.3.1 Variable Selection](#2_3_1-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.3.2 Feature Contribution](#2_3_2-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.3.3 Interaction Strengths](#2_3_3-bullet) <br>
<br>
[__3. Datensatz 2: freMTPL2freq__](#3-bullet)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1 Vorverarbeitung](#3_1-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2 LocalGLMnet](#3_2-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.3 Auswertung](#3_3-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.3.1 Variable Selection](#3_3_1-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.3.2 Neues LocalGLMnet trainieren](#3_3_2-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.3.3 Feature Contribution](#3_3_3-bullet) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.3.4 Interaction Strengths](#3_3_4-bullet) <br>
<br>
[__4. Zusammenfassung__](#4-bullet)<br>
<br>
[__Literaturverzeichnis__](#literatur)

# Abbildungsverzeichnis <a class="anchor" id="abbildungsverzeichnis"></a>

__Synthetischer Datensatz:__<br>
[__Abb. 1:__ LocalGLMnet vs. GLM](#abb1) <br>
[__Abb. 2:__ Regression Attentions](#abb2) <br>
[__Abb. 3:__ Feature Contributions](#abb3) <br>
[__Abb. 4:__ Interaction Strengths](#abb4) <br>
<br>
__FreMTPL-Datensatz:__<br>
[__Abb. 5:__ Regression Attentions](#abb5) <br>
[__Abb. 6:__ Area Code vs. Density](#abb6) <br>
[__Abb. 7:__ Feature Contributions](#abb7) <br>
[__Abb. 8:__ Feature Contribution kategorialer Variablen](#abb8) <br>
[__Abb. 9:__ Interaction Strengths](#abb9) <br>

# Dependencies <a class="anchor" id="dependencies"></a>

<font size="3">
Zur Ausführung des Codes in diesem Notebook sind die folgenden Bibliotheken notwendig:


```python
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from scipy import interpolate
import scipy.stats as stats
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from labellines import labelLines
import os
import random

```

<font size="3">
Ein entsprechendes Anaconda Environment lässt sich mithilfe der <em>requirements.txt</em>-Datei und dem conda-Befehl <em>conda create --name &lt;env&gt; --file &lt;requirements.txt&gt;></em> installieren.<br>
<br>
Um reproduzierbare Ergebnisse zu gewährleisten, werden zusätzlich die Zufallsgeneratoren mit dem Seed 0 initialisiert.


```python
# Seed der Zufallsgeneratoren festlegen
seed = 0
tf.random.set_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

```

<hr>

# 1. Einleitung <a class="anchor" id="1-bullet"></a>

<hr>

<font size="3">
In dem Paper <em>LocalGLMnet: interpretable deep learning for tabular data</em> beschreiben Richman & Wüthrich eine neue Struktur für Neuronale Netze, welche auf Generalisierten Linearen Modellen (GLMs) beruht [1]. Dies soll einen Kompromiss zwischen der hohen Performanz der klassischen vorwärts gerichteten Neuronalen Netzen und der Erklärbarkeit von GLMs schaffen. Der Grundgedanke besteht darin, dass die Koeffizienten des GLMs durch das Neuronale Netz bestimmt werden. Hierdurch können sie anders als bei einem klassischen GLM in Abhängigkeit von den Merkmalsausprägungen variieren. Für einen beschränkten Wertebereich können die Koeffizienten jedoch konstant erscheinen, weshalb von einem lokalen GLM gesprochen wird.
Um den Zusammenhang zwischen Attention-Gewichten zu den ursprünglichen Merkmalen beizubehalten und Rückschlüsse auf den Einfluss unterschiedlicher Merkmale zu erlauben, wird in der Netzstruktur eine Skip-Connection verwendet. Bevor die Attention-Gewichte als Parameter des GLMs verwendet werden, werden sie hierfür mit der entsprechenden ursprünglichen Merkmalsausprägung multipliziert. <br>
<br>
Im Folgenden soll die Modellierung des LocalGLMnet-Ansatzes in Python mithilfe der Tensorflow Implementierung der Keras API dargestellt werden. Hierfür werden dieselben Datensätze wie im ursprünglichen Paper verwendet, um einen einfachen Transfer der Inhalte des Papers zur Implementierung zu ermöglichen. Der erste Datensatz ist hierbei ein synthetischer Datensatz, der dadurch, dass der tatsächliche Regressionszusammenhang bekannt ist, Möglichkeiten sowie Grenzen des Ansatzes aufzeigt. Anschließend wird der Einsatz des LocalGLMnet an einem realen Sachverhalt, der Vorhersage der Schadensmeldungen einer Kfz-Haftpflicht, dargestellt.

<hr>

# 2. Datensatz 1: Künstlicher Datensatz <a class="anchor" id="2-bullet"></a>

<hr>

## 2.1 Künstlichen Datensatz erzeugen <a class="anchor" id="2_1-bullet"></a>

<font size="3">
Der synthetische Datensatz besteht aus insgesamt 8 Merkmalen. <em>x7</em> und <em>x8</em> haben keinen Einfluss auf die Zielvariable. <em>x8</em> ist jedoch zu 50% mit <em>x2</em> korreliert. Der funktionale Zusammenhang der Zielvariable ergibt sich wie folgt:

\begin{equation} 
\mu\left( x \right)=\frac{1}{2}x_{1}-\frac{1}{4}x^2_{2}+\frac{1}{2}\left\lvert x_{3} \right\rvert sin\left( 2x_{3} \right)+\frac{1}{2}x_{4}x_{5}+\frac{1}{8}x^{2}_{5}x_{6}
\end{equation}

Die Merkmalsausprägungen werden mithilfe des Zufallsgenerators von Numpy auf Basis einer Standardnormalverteilung erzeugt. Hierdurch sind die Merkmale bereits standardisiert, d. h. alle haben den Mittelwert <em>µ=0</em> und <em>std=1</em>. Bei einem anderen Datensatz müssten die Merkmale zuerst standardisiert werden, damit die Werte die gleiche Größenordnung haben. Da die Daten künstlich erzeugt werden, wird sowohl ein Trainings- als auch ein Testdatensatz mit 100000 Beobachtungen erzeugt. Bei einem realen Datensatz müsste der vorhandene Datensatz entsprechend aufgeteilt werden (bspw. 80:20).


```python
# Zielfunktion
def target_variable(x):
    return (
        (1 / 2) * x[0]
        - (1 / 4) * (x[1] ** 2)
        + (1 / 2) * abs(x[2]) * math.sin(2 * x[2])
        + (1 / 2) * x[3] * x[4]
        + (1 / 8) * (x[4] ** 2) * x[5]
    )

```


```python
# Random Number Generator
rng = np.random.default_rng()

# Trainingsdatensatz (n = 100.000) erzeugen (Variablen x1, x3, x4, x5, x6, x7)
x1_train = rng.standard_normal(size=(100000, 1))
x3_7_train = rng.standard_normal(size=(100000, 5))

# Variablen x2, x8 mit 50 % Korrelation erzeugen
cov_matrix = [[1, 0.5], [0.5, 1]]
x2_x8_train = rng.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=100000)

# Trainingsdatensatz zusammenfügen und Zielvariable y bestimmen
x_train = np.column_stack((x1_train, x2_x8_train[:, 0], x3_7_train, x2_x8_train[:, 1]))
y_train = np.array(list(map(target_variable, x_train[:, 0:7])))

```


```python
# Testdatensatz (n = 100.000) erzeugen (Variablen x1, x3, x4, x5, x6, x7)
x1_test = rng.standard_normal(size=(100000, 1))
x3_7_test = rng.standard_normal(size=(100000, 5))

# Variablen x2, x8 mit 50 % Korrelation erzeugen
x2_x8_test = rng.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=100000)

# Testdatensatz zusammenfügen und Zielvariable y bestimmen
x_test = np.column_stack((x1_test, x2_x8_test[:, 0], x3_7_test, x2_x8_test[:, 1]))
y_test = np.array(list(map(target_variable, x_test[:, 0:7])))

```

## 2.2 LocalGLMnet <a class="anchor" id="2_2-bullet"></a>

<font size="3">
In diesem Kapitel soll auf Basis der Trainingsdaten zuerst ein klassisches GLM und anschließend ein LocalGLMnet trainiert werden. Für diese bietet sich daraufhin der Vergleich der Vorhersagegenauigkeit beider Modelle an.

### 2.2.1 GLM <a class="anchor" id="2_2_1-bullet"></a>

<font size="3">
Generalisierte Lineare Modelle, kurz GLM, erweitern klassische lineare Regressionsfunktionen, sodass auch nichtlineare Zusammenhänge abgebildet werden können [2]. Grundsätzlich bestehen hierfür GLMs aus drei Komponenten: der systematischen Komponente, der Link-Funktion und der Zufallskomponente.
Bei der systematischen Komponente handelt es sich um die klassischen Regressionsparameter einer linearen Funktion. Die Zufallskomponente beschreibt die Verteilung der Residuen einer klassischen linearen Regression. Während diese bei einer klassischen linearen Regression normalverteilt sein müssen, erlaubt ein GLM andere Verteilungen aus der Exponentialfamilie wie die Binomial- oder Poisson-Verteilung. Die Link-Funktion verbindet die Regressionsparameter mit der spezifizierten Zufallskomponente. Abhängig von dem Problem, für das ein GLM eingesetzt werden soll und den zugrundeliegenden Daten bieten sich unterschiedliche Zufallskomponenten und entsprechende Link-Funktionen an. Genaueres zu der Wahl der Link-Funktion findet sich bspw. in [2]. Unter anderem in Versicherungen werden GLMs vielseitig eingesetzt, da ihre Ergebnisse anhand der Koeffizienten gut interpretierbar sind.
<br><br>
Für den synthetischen Datensatz haben Richman & Wüthrich die Identity-Link Funktion verwendet. Das resultierende GLM entspricht also einer klassischen linearen Regression. Um ein GLM mit Python zu erzeugen, bieten sich Bibliotheken wie <em>scikit-learn</em> oder <em>statsmodels</em> an.


```python
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

```




    LinearRegression()



### 2.2.2 LocalGLMnet <a class="anchor" id="2_2_2-bullet"></a>

<font size="3">
Entsprechend der Spezifikationen in Richman & Wüthrich [1] wurde ebenfalls ein LocalGLMnet trainiert. Es besitzt vier Hidden Layer.
Die Anzahl der zu erlernenden Attention-Gewichte entspricht der Anzahl der Inputvariablen. Um die Skip-Connection zu realisieren, wird dem Netz eine weitere Ebene hinzugefügt, welche das Skalarprodukt des Inputs und der Attention-Gewichte bildet. Um ebenfalls den Intercept <em>β0</em>  abbilden zu können besitzt das Netz eine weitere Schicht mit einem Neuron. Durch das Anpassen der Aktivierungsfunktion lassen sich verschiedene Link-Funktionen implementieren. Das LocalGLMnet speißt die Attention-Gewichte in der letzten Schicht also direkt in das resultierende lokale GLM ein.


```python
# LocalGLMnet strukturieren

input = tf.keras.Input(shape=(8), dtype="float32")

attention = input
attention = tf.keras.layers.Dense(units=20, activation="tanh")(attention)
attention = tf.keras.layers.Dense(units=15, activation="tanh")(attention)
attention = tf.keras.layers.Dense(units=10, activation="tanh")(attention)
attention = tf.keras.layers.Dense(units=8, activation="linear", name="Attention")(
    attention
)

# Skip-Connection
response = tf.keras.layers.Dot(axes=1)([input, attention])

# Response Schicht = lokales GLM
response = tf.keras.layers.Dense(units=1, activation="linear", name="Response")(
    response
)

```

    2023-02-19 13:13:06.119452: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
# Modell kompilieren
local_glm_net = tf.keras.Model(inputs=input, outputs=response)
local_glm_net.compile(loss="mse", optimizer="nadam")
local_glm_net.summary()

```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 8)]          0                                            
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 20)           180         input_1[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 15)           315         dense[0][0]                      
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 10)           160         dense_1[0][0]                    
    __________________________________________________________________________________________________
    Attention (Dense)               (None, 8)            88          dense_2[0][0]                    
    __________________________________________________________________________________________________
    dot (Dot)                       (None, 1)            0           input_1[0][0]                    
                                                                     Attention[0][0]                  
    __________________________________________________________________________________________________
    Response (Dense)                (None, 1)            2           dot[0][0]                        
    ==================================================================================================
    Total params: 745
    Trainable params: 745
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
# Modell trainieren
history = local_glm_net.fit(
    x_train, y_train, batch_size=32, epochs=10, validation_split=0.2
)

```

    2023-02-19 13:13:06.404522: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)


    Epoch 1/10
    2500/2500 [==============================] - 4s 1ms/step - loss: 0.2422 - val_loss: 0.0840
    Epoch 2/10
    2500/2500 [==============================] - 3s 1ms/step - loss: 0.0612 - val_loss: 0.0480
    Epoch 3/10
    2500/2500 [==============================] - 4s 2ms/step - loss: 0.0391 - val_loss: 0.0470
    Epoch 4/10
    2500/2500 [==============================] - 4s 2ms/step - loss: 0.0295 - val_loss: 0.0257
    Epoch 5/10
    2500/2500 [==============================] - 3s 1ms/step - loss: 0.0155 - val_loss: 0.0092
    Epoch 6/10
    2500/2500 [==============================] - 3s 1ms/step - loss: 0.0074 - val_loss: 0.0058
    Epoch 7/10
    2500/2500 [==============================] - 3s 1ms/step - loss: 0.0051 - val_loss: 0.0050
    Epoch 8/10
    2500/2500 [==============================] - 3s 1ms/step - loss: 0.0043 - val_loss: 0.0046
    Epoch 9/10
    2500/2500 [==============================] - 3s 1ms/step - loss: 0.0037 - val_loss: 0.0035
    Epoch 10/10
    2500/2500 [==============================] - 3s 1ms/step - loss: 0.0037 - val_loss: 0.0038


### 2.2.3 Performance Vergleich <a class="anchor" id="2_2_3-bullet"></a>

<font size="3">
Bei einem Vergleich der Vorhersagegenauigkeit schneidet das LocalGLMnet deutlich besser ab als ein klassisches GLM. Dargestellt in Abbildung 1 wird gut ersichtlich, dass das LocalGLMnet den funktionalen Zusammenhang im Vergleich zum GLM sehr gut abbilden kann. Die Vorhersagen des LocalGLMnet haben einen MSE von gerade einmal ≈0,003 während das GLM einen MSE von ≈0,533 aufweist. Dies ist jedoch durch die generell hohe Vorhersagegenauigkeit von Neuronalen Netzen nicht verwunderlich. Wichtig ist, dass nichtsdestotrotz die Interpretierbarkeit der Vorhersagen bestehen bleibt. Dies wird im nächsten Abschnitt behandelt.

<a class="anchor" id="abb1"></a>


```python
# Vorhersage mit localGLMnet und GLM
pred_local = local_glm_net.predict(x_test)
pred_reg = reg.predict(x_test)

fig_performance = plt.figure(tight_layout=True, figsize=(10, 5))

spec = GridSpec(ncols=2, nrows=1, figure=fig_performance)
axs_perf = [
    fig_performance.add_subplot(spec[0, 0:1]),
    fig_performance.add_subplot(spec[0, 1:2]),
]

axs_perf[0].scatter(y_test, pred_local, s=1)
axs_perf[1].scatter(y_test, pred_reg, s=1)

# Layout
for ax in axs_perf:
    ax.set_xlabel("True value")
    ax.set_ylabel("Estimated value")
    ax.set_xlim((-4, 4))
    ax.set_ylim((-4, 4))

fig_performance.suptitle("Abbildung 1: LocalGLMnet vs. GLM")
plt.show()

print("MSE LocalGLMnet: " + str(metrics.mean_squared_error(y_test, pred_local)))
print("MSE GLM: " + str(metrics.mean_squared_error(y_test, pred_reg)))

```


    
![png](/src/localGLMnet_markdown_assets/localGLMnet_30_0.png)
    


    MSE LocalGLMnet: 0.0037230475765324548
    MSE GLM: 0.5342545394609545


## 2.3 Auswertung <a class="anchor" id="2_3-bullet"></a>

### 2.3.1 Variable Selection <a class="anchor" id="2_3_1-bullet"></a>

<font size="3">
Ein Vorteil von LocalGLMnets gegenüber klassischen FFNs ist, dass sie sich zur Merkmalsselektion eignen.
Um einen ersten Eindruck vom Einfluss der Attention-Gewichte auf die Vorhersage sowie dem Zusammenhang zur ursprünglichen Inputvariable zu erhalten, bietet sich ein Scatterplot der Regressionsgewichte in Abhängigkeit der Inputvariablen an. Geringe laterale Verzerrungen bedeuten, dass es wenig Interaktionen des Attention-Gewichts mit anderen Inputvariablen gibt. Gleichzeitig lassen sich gewisse funktionale Zusammenhänge bereits durch den Plot erkennen. <br><br>
Streut ein Attention-Wert für einen Großteil des Wertebereichs der Inputvariable um 0, scheint der Einfluss vernachlässigbar, das Merkmal kann demnach entfernt werden. Um ein Maß für die Streuung zu bieten, haben Richman & Wüthrich einen empirischen Wald-Test entwickelt [1]. Hierbei wird dem Modell eine zusätzliche Variable ohne Zusammenhang zur Zielvariable hinzugefügt. Anschließend wird auf Basis der Streuung des zugehörigen Attention-Gewichts ein Konfidenzintervall berechnet. Hiermit lässt sich daraufhin die Coverage Ratio für jedes Attention-Gewicht, also der Anteil der Gewichte, die innerhalb der Grenzen liegen, berechnen. Ist diese kleiner als das Signifikanzniveau, kann die Variable entfernt und das Modell erneut ohne diese trainiert werden.
<br><br>
Bei dem verwendeten synthetischen Datensatz kann direkt <em>β7</em> verwendet werden, da sie keinen Einfluss auf die tatsächliche Regressionsfunktion hat. Bei einem realen Datensatz bieten sich künstlich erzeugte normal- und gleichverteilte Merkmale mit μ=0 und std=1 an.


```python
# Über die Methode get_weights() erhält man die Kantengewichte, sowie den Bias für jeder Schicht
# --> man erhält also eine Liste mit numpy Arrays die in der Länge der Anzahl der Ebenen * 2 entspricht
weights = local_glm_net.get_weights()
for i in weights:
    print(i.shape, end=" | ")

```

    (8, 20) | (20,) | (20, 15) | (15,) | (15, 10) | (10,) | (10, 8) | (8,) | (1, 1) | (1,) | 


```python
# Neues Model ohne Response-Schicht --> ermöglicht auslesen der Attention Gewichte
weights_local_glm = tf.keras.Model(
    inputs=local_glm_net.input, outputs=local_glm_net.get_layer(name="Attention").output
)

# Gewichte bestimmen
beta_x = weights_local_glm.predict(x_test)

# Skalierung der Attention-Gewichte mithilfe des Gewichts der Response Schicht ( = Intercept beta_0)
beta_x_scaled = beta_x * weights[8]

```


```python
# Merkmal 7 ist von der wahren Regressionsfunktion unabhängig
# --> Einsatz zur Berechnung des Konfidenzintervals

print("Mittelwert β7: " + str(beta_x_scaled[:, 6].mean()))
print("Standardabweichung β7: " + str(beta_x_scaled[:, 6].std()))

# Intervalgrenzen bestimmen
alpha = 0.001
bound = stats.norm.ppf(alpha / 2) * beta_x_scaled[:, 6].std()

print("Quantil " + str(1 - alpha / 2) + ": " + str(stats.norm.ppf(alpha / 2)))
print("Grenzen: ± " + str(abs(bound)))

```

    Mittelwert β7: -0.003177547
    Standardabweichung β7: 0.008993691
    Quantil 0.9995: -3.2905267314918945
    Grenzen: ± 0.029593980102238748


<a class="anchor" id="abb2"></a>


```python
# Attention Plot
fig_attention = plt.figure(tight_layout=True, figsize=(30, 15))

# Gliederung der Subplots
spec = GridSpec(ncols=8, nrows=3, figure=fig_attention)
ax1_att = fig_attention.add_subplot(spec[0, 1:3])
ax2_att = fig_attention.add_subplot(spec[0, 3:5])
ax3_att = fig_attention.add_subplot(spec[0, 5:7])
ax4_att = fig_attention.add_subplot(spec[1, 1:3])
ax5_att = fig_attention.add_subplot(spec[1, 3:5])
ax6_att = fig_attention.add_subplot(spec[1, 5:7])
ax7_att = fig_attention.add_subplot(spec[2, 2:4])
ax8_att = fig_attention.add_subplot(spec[2, 4:6])
axs_att = [ax1_att, ax2_att, ax3_att, ax4_att, ax5_att, ax6_att, ax7_att, ax8_att]

# Ein Subplot pro Input Feature erstellen
for i in range(len(axs_att)):

    # Linien zur Verdeutlichung der Höhe der Attention Gewichte
    axs_att[i].hlines(y=0.5, xmin=-4, xmax=4, colors="orange")
    axs_att[i].hlines(y=-0.5, xmin=-4, xmax=4, colors="orange")

    axs_att[i].hlines(y=0.25, xmin=-4, xmax=4, colors="orange", linestyles="dashed")
    axs_att[i].hlines(y=-0.25, xmin=-4, xmax=4, colors="orange", linestyles="dashed")

    axs_att[i].hlines(y=0, xmin=-4, xmax=4, colors="red")

    # Intervalgrenzen
    interval = patches.Rectangle(
        xy=(-4, bound),
        height=2 * abs(bound),
        width=8,
        edgecolor="royalblue",
        facecolor="lightcyan",
        alpha=0.8,
        zorder=1,
    )
    axs_att[i].add_patch(interval)

    # Scatter Plot --> x: Werte der Inputfeatures, y: Attention Gewichte
    axs_att[i].scatter(x_test[:, i], beta_x_scaled[:, i], s=0.5, c="black")

    # Layout
    axs_att[i].set_xlim((-4, 4))
    axs_att[i].set_ylim((-1, 1))
    axs_att[i].set_xlabel("x" + str(i + 1))
    axs_att[i].set_ylabel("Attention β" + str(i + 1))

fig_attention.suptitle("Abbildung 2: Regression Attentions")
plt.show()

```


    
![png](/src/localGLMnet_markdown_assets/localGLMnet_38_0.png)
    


<font size="3">
<b>In dem Plot lassen sich bereits verschiedene funktionale Zusammenhänge und Interaktionen erkennen:</b><br>
<ul>
    <li><b>β1</b></li>
        <ul>
          <li>liegt relativ konstant bei 0.5</li>
          <li>wenig laterale Verzerrungen --> kaum Interaktionen mit anderen Inputvariablen</li>
        </ul><br>
    <li><b>β2, β3</b></li>
        <ul>
          <li>wenig laterale Verzerrungen --> kaum Interaktionen mit anderen Inputvariablen</li>
        </ul><br>
    <li><b>β4, β5, β6</b></li>
        <ul>
          <li>Attention-Gewicht ≠ 0 --> Einfluss auf die Vorhersage</li>
          <li>laterale Verzerrungen --> Interaktionen mit anderen Inputvariablen</li>
        </ul><br>
    <li><b>β7, β8</b></li>
        <ul>
          <li>streuen um 0 --> β8 streut durch die Korrelation von x2 und x8 stärker (durch Wald-Test muss entschieden werden, ob x8 entfernt werden kann) </li>
        </ul>
</ul>


```python
for i in range(8):
    if i != 6:
        size = beta_x_scaled.shape[0]
        coverage = np.count_nonzero(
            beta_x_scaled[:, i] < abs(bound)
        ) - np.count_nonzero(beta_x_scaled[:, i] < -abs(bound))
        coverage_ratio = coverage / size
        print("Coverage Ratio β" + str(i + 1) + ": " + str(coverage_ratio))

```

    Coverage Ratio β1: 0.0
    Coverage Ratio β2: 0.09149
    Coverage Ratio β3: 0.01268
    Coverage Ratio β4: 0.09206
    Coverage Ratio β5: 0.086
    Coverage Ratio β6: 0.22275
    Coverage Ratio β8: 0.57559


<font size="3">
<b>Es kann keine Inputvariable entfernt werden, da die Coverage Ratio von keinem Attention-Gewicht ≥ 99,9% ist.</b> <br>
<em>Im Paper wird β8 entfernt, da die Coverage Ratio über 0.999 liegt. Schwankungen können bspw. durch die zufällig erzeugten Daten oder die für einen Trainingsbatch ausgewählten Beobachtungen verursacht werden.</em>

### 2.3.2 Feature Contribution <a class="anchor" id="2_3_2-bullet"></a>

<font size="3">
Bevor die Attention Gewichte als Parameter des GLMs verwendet werden, wird das Skalarprodukt mit der ursprünglichen Inputvariable gebildet. Die resultierende Größe ist die Feature Contribution. Eine Visualisierung dieser in Abhängigkeit von der Inputvariable zeigt deutlicher den resultierenden funktionalen Zusammenhang. Zur Verdeutlichung können zusätzlich Splines hinzugefügt werden, welche diesen approximieren.

<a class="anchor" id="abb3"></a>


```python
# Feature Contribution Plot
fig_contribution = plt.figure(tight_layout=True, figsize=(30, 15))

spec = GridSpec(ncols=8, nrows=3, figure=fig_contribution)
ax1_con = fig_contribution.add_subplot(spec[0, 1:3])
ax2_con = fig_contribution.add_subplot(spec[0, 3:5])
ax3_con = fig_contribution.add_subplot(spec[0, 5:7])
ax4_con = fig_contribution.add_subplot(spec[1, 1:3])
ax5_con = fig_contribution.add_subplot(spec[1, 3:5])
ax6_con = fig_contribution.add_subplot(spec[1, 5:7])
ax7_con = fig_contribution.add_subplot(spec[2, 2:4])
ax8_con = fig_contribution.add_subplot(spec[2, 4:6])

axs_con = [ax1_con, ax2_con, ax3_con, ax4_con, ax5_con, ax6_con, ax7_con, ax8_con]

xs = np.linspace(-4, 4, 1000)

for i in range(len(axs_con)):

    # Feature Contribution Splines berechnen
    # Feature Contribution = beta(xi)*xi
    contribution = np.column_stack([x_test[:, i], beta_x_scaled[:, i] * x_test[:, i]])
    con_ind = np.lexsort((contribution[:, 1], contribution[:, 0]))
    contribution_sorted = contribution[con_ind]
    con_spline = interpolate.UnivariateSpline(
        contribution_sorted[:, 0], contribution_sorted[:, 1]
    )

    # Hinzufügen von horizontalen Linien um die Stärke der Feature Contribution zu visualisieren
    axs_con[i].hlines(y=0, xmin=-4, xmax=4, colors="orange", alpha=0.7, zorder=1)

    axs_con[i].hlines(y=0.5, xmin=-4, xmax=4, colors="red", alpha=0.5, zorder=1)
    axs_con[i].hlines(y=-0.5, xmin=-4, xmax=4, colors="red", alpha=0.5, zorder=1)

    axs_con[i].hlines(y=1, xmin=-4, xmax=4, colors="lightcyan", alpha=0.7, zorder=1)
    axs_con[i].hlines(y=-1, xmin=-4, xmax=4, colors="lightcyan", alpha=0.7, zorder=1)

    axs_con[i].hlines(y=1.5, xmin=-4, xmax=4, colors="royalblue", alpha=0.7, zorder=1)
    axs_con[i].hlines(y=-1.5, xmin=-4, xmax=4, colors="royalblue", alpha=0.7, zorder=1)

    # Scatter Plot --> x: Werte der Inputfeatures, y:Feature Contribution (β(x)*x)
    axs_con[i].scatter(contribution[:, 0], contribution[:, 1], s=0.5, zorder=10)

    # Feature Contribution Spline plotten
    axs_con[i].plot(xs, con_spline(xs), color="purple", zorder=20)

    # Layout
    axs_con[i].set_xlim((-4, 4))
    axs_con[i].set_ylim((-2, 2))
    axs_con[i].set_xlabel("x" + str(i + 1))
    axs_con[i].set_ylabel("Feature Contribution (beta(x)*x)" + str(i + 1))

fig_contribution.suptitle("Abbildung 3: Feature Contributions")
plt.show()

```


    
![png](/src/localGLMnet_markdown_assets/localGLMnet_45_0.png)
    


<font size="3">
<ul>
    <li><b>β1</b></li>
        <ul>
          <li>lineare Funktion entsprechend der zugrundeliegenden Regressionsfunktion (<b>&frac12;x<sub>1</sub></b>)</li>
          <li>wenig laterale Verzerrungen --> kaum Interaktionen mit anderen Inputvariablen</li>
        </ul><br>
    <li><b>β2</b></li>
        <ul>
          <li>quadratischer Zusammenhang erkennbar (<b>&frac14;x<sub>2</sub><sup>2</sup></b>)</li>
          <li>wenig laterale Verzerrungen --> kaum Interaktionen mit anderen Inputvariablen</li>
        </ul><br>
    <li><b>β3</b></li>
        <ul>
          <li>Sinusfunktion (<b>&frac12; |x<sub>3</sub>| sin(2x<sub>3</sub>)</b>)</li>
          <li>wenig laterale Verzerrungen --> kaum Interaktionen mit anderen Inputvariablen</li>
        </ul><br>
    <li><b>β4, β5, β6</b></li>
        <ul>
          <li>laterale Verzerrungen --> starke Interaktionen mit anderen Inputvariablen</li>
        </ul><br>
    <li><b>β7, β8</b></li>
        <ul>
          <li> streuen um 0 --> β8 weist etwas mehr Interaktionen auf </li>
        </ul>
</ul>

### 2.3.3 Interaction Strengths <a class="anchor" id="2_3_3-bullet"></a>

<font size="3">

Um die zuvor bereits erkannten Interaktionen genauer zu analysieren, bietet es sich an, die Gradienten der Attention Gewichte zu untersuchen. Liegt keine Interaktion zwischen einem Attention-Gewicht $j$ und einer Inputvariable $j'$ vor, so ist $∂β_j(x)/∂x_{j'}=0$.
<br>
Zur Darstellung der Gradienten bieten sich Regressionssplines an. Bei diesen handelt es sich um eine aus mehreren Polynomen zusammengesetzte Funktion, welche daher besonders "glatt" verläuft [3].
<br><br>
Im Paper verwenden Richman & Wüthrich die R Bibliothek <em>locfit</em> [1]. Da diese nicht für Python verfügbar ist muss auf eine andere Bibliothek ausgewichen werden. Eine Möglichkeit zur Modellierung eines Univariaten Splines bietet bspw. <em>scipy</em>. Dies entspricht nicht genau der Implementierung mittels <em>locfit</em>, ermöglicht jedoch die gleichen Schlüsse.


```python
# Gradienten bestimmen
gradients = []
x = tf.constant(x_train)

# Für jede Inputvariable wird ein Modell gefittet, um anschließend die partiellen Ableitungen auslesen zu können
for i in range(input.shape[-1]):

    # Lambda Layer als Output Schicht, um beta_i als Output zu erhalten (partielle Ableitungen ∂β_j(x)/∂x_j')
    beta = attention
    beta = tf.keras.layers.Lambda(lambda x: x[:, i])(beta)
    grad_model = tf.keras.Model(inputs=input, outputs=beta)

    # GradientTape ermöglicht das auslesen der Gradienten
    with tf.GradientTape() as g:
        g.watch(x)
        pred_attention = grad_model.call(x)

    grad = g.gradient(pred_attention, x)

    # Array das sowohl den Wert von x, als auch den entsprechenden Wert von βk(x) enthält
    grad_wrt_x = np.column_stack((x[:, i].numpy(), grad.numpy()))

    # Um später die Splines zu modellieren muss die x-Komponente monoton steigend sein --> sortieren des Arrays
    ind = np.lexsort((grad_wrt_x[:, 2], grad_wrt_x[:, 0]))
    grad_wrt_x_sorted = grad_wrt_x[ind]

    # Gradienten in Liste speichern
    gradients.append(grad_wrt_x_sorted)

```


```python
# Univariate Splines modellieren, um die Interaktion zwischen Features darzustellen
splines = []

# Für alle Attention Gewichte β
for i in range(input.shape[-1]):
    splines.append([])

    # Für alle Inputvariablen x
    for j in range(input.shape[-1]):
        splines[i].append(
            interpolate.UnivariateSpline(gradients[i][:, 0], gradients[i][:, j + 1])
        )

```

<a class="anchor" id="abb4"></a>


```python
# Spline Interaction Plot

fig_spline = plt.figure(tight_layout=True, figsize=(30, 15))
spec = GridSpec(ncols=8, nrows=3, figure=fig_spline)
ax1_sp = fig_spline.add_subplot(spec[0, 1:3])
ax2_sp = fig_spline.add_subplot(spec[0, 3:5])
ax3_sp = fig_spline.add_subplot(spec[0, 5:7])
ax4_sp = fig_spline.add_subplot(spec[1, 1:3])
ax5_sp = fig_spline.add_subplot(spec[1, 3:5])
ax6_sp = fig_spline.add_subplot(spec[1, 5:7])
ax7_sp = fig_spline.add_subplot(spec[2, 2:4])
ax8_sp = fig_spline.add_subplot(spec[2, 4:6])

axs_sp = [ax1_sp, ax2_sp, ax3_sp, ax4_sp, ax5_sp, ax6_sp, ax7_sp, ax8_sp]

xs = np.linspace(-4, 4, 100)

for i in range(input.shape[-1]):

    # Splines für jedes Merkmal plotten
    for j in range(input.shape[-1]):
        axs_sp[i].plot(xs, splines[i][j](xs), label="x" + str(j + 1))

    # Inline Lables und Legende hinzufügen
    labelLines(axs_sp[i].get_lines(), zorder=2.5)
    axs_sp[i].legend(loc="lower right", ncol=2)

    # Layout
    axs_sp[i].set_xlim((-4, 4))
    axs_sp[i].set_ylim((-0.5, 0.5))
    axs_sp[i].set_xlabel("Feature Values x" + str(i + 1))
    axs_sp[i].set_ylabel("Interaction Strengths")
    axs_sp[i].set_title("Interactions of Feature Component x" + str(i + 1))
fig_spline.suptitle("Abbildung 4: Interaction Strengths")

plt.show()

```


    
![png](/src/localGLMnet_markdown_assets/localGLMnet_52_0.png)
    


<font size="3">
<b>Interaktionen:</b>
<ul>
    <li><b>x1, x6, x7, x8</b></li>
        <ul>
          <li> Der Wert aller partiellen Ableitungen liegt konstant bei ≈0 </li>
          <li> → Keine Interaktionen (β ist konstant)</li>
        </ul><br>
    <li><b>x2</b></li>
        <ul>
          <li> Großteil der Werte um 0 konzentriert </li>
          <li> x2 ≠ 0</li>
          <li> → Sehr geringe Interaktionen mit anderen Variablen, allerdings nicht-linearer Zusammenhang mit x2 (quadratisch)</li>
        </ul><br>
    <li><b>x3</b></li>
        <ul>
          <li> generell um 0 zentriert, aber größere Streuung als bei x2</li>
          <li> x3 ≠ 0 → Sinus ähnelndes Verhalten </li>
          <li> → geringe Interaktionen mit anderen Variablen</li>
        </ul><br>
    <li><b>x4</b></li>
        <ul>
          <li> lineare Interaktion mit x5 (const. ≈ 0.3)</li>
          <li> → Sehr geringe Interaktionen mit anderen Variablen</li>
        </ul><br>
    <li><b>x5</b></li>
        <ul>
          <li> Geringe Interaktionen mit anderen Variablen </li>
          <li> Stärkste Interaktionen mit x4 (linear) und x5</li>
        </ul>
</ul>

<hr>

# 3. Datensatz #2: freMTPL2freq <a class="anchor" id="3-bullet"></a>
<hr>

<font size="3">

Nachdem der Ansatz des LocalGLMnet grundlegend anhand des synthetischen Datensatzes vorgestellt wurde, soll dieser nun auf einen realen Sachverhalt angewendet werden. Richman und Wüthrich verwenden hierfür den freMTPL (= French Motor Third-Part Liability) Datensatz, da er generell als Benchmark im Aktuarsbereich gilt [1]. Der Datensatz enthält Informationen über Kfz-Haftpflichtversicherungen und aufgetretene Schadensmeldungen.

## 3.1 Vorverarbeitung <a class="anchor" id="3_1-bullet"></a>

<font size="3">
Um ähnliche Ergebnisse wie Richman und Wüthrich [1] zu erhalten, wurden die gleichen Schritte zur Vorverarbeitung des Datensatzes durchgeführt. Diese werden in Wüthrich/Merz [4] genauer dargestellt. <br>
Im ersten Schritt werden die Datensätze FreMTPL2freq und FreMTPL2sev zusammengeführt. FreMTPL2freq enthält Informationen über die Versicherungspolicen und FreMTPL2sev über die aufgetretenen Schäden. FreMTPL2freq enthält zwar ebenfalls die Schadensanzahl, jedoch scheint es hierbei einige inkorrekte Aufzeichnungen zu geben. Eine Erläuterung der sonstigen Merkmale lässt sich Wüthrich/Merz [4, S. 555] entnehmen:<br>
<ol>
    <li><b>IDpol:</b> policy number (unique identiﬁer)</li>
    <li><b>Exposure:</b> total exposure in yearly units (years-at-risk) and within (0, 1 ]</li>
    <li><b>Area:</b> area code (categorical, ordinal with 6 levels)</li>
    <li><b>VehPower:</b> power of the car (continuous)</li>
    <li><b>VehAge</b> age of the car in years</li>
    <li><b>DrivAge:</b> age of the (most common) driver in years</li>
    <li><b>BonusMalus:</b> bonus-malus level between 50 and 230 (with entrance level 100)</li>
    <li><b>VehBrand:</b> car brand (categorical, nominal with 11 levels)</li>
    <li><b>VehGas:</b> diesel or regular fuel car (binary)</li>
    <li><b>Density:</b> density of population per km 2 at the location of the living place of the driver</li>
    <li><b>Region:</b> regions in France (prior to 2016)(categorical)</li>
</ol>
<br>
Entsprechend der Anweisungen von Merz/Wüthrich [4] wurde der FreMTPL2freq Datensatz in der Version 1.0-8 über die <a href="https://www.openml.org/search?type=data&sort=runs&id=41214&status=active">OpenML ID 41214</a> heruntergeladen. Dennoch entspricht die Anzahl der Kategorien von VehBrand mit 14 nicht der Anzahl im Paper. Aus diesem Grund werden im Folgenden leichte Anpassungen vorgenommen, bspw. hat das LocalGLMnet hierdurch eine Inputdimension q=45. <br>
<br>
Nachdem der Datensatz in ein DataFrame geladen wurde, wird den Merkmalen der zugehörige Datentyps zugeordnet und teils weitere Vorverarbeitungen vorgenommen. Der maximale Wert der <em>Exposures</em> wurde bspw. auf 1 begrenzt, da lediglich betrachtet wird, ob die Policen im ganzen Jahr aktiv sind. Beobachtungen mit mehr als 5 Schadensfällen werden zudem entfernt, da es sich hierbei höchstwahrscheinlich um fehlerhafte Daten handelt. Um den Einfluss der kategorialen Variablen <em>VehBrand</em> und <em>Region</em> im Modell abbilden zu können, werden diese mittels One-Hot Encoding transformiert. Um später ein Maß für die zufälligen Schwankungen der Attention-Gewichte zu haben, wird eine gleichverteilte (= <em>RandU</em>) und eine normalverteilte (= <em>RandN</em>) Störvariable hinzugefügt<br>
Abschließend werden die Daten in Trainings- und Testdatensätze mit einem Split von 90:10 aufgeteilt und so skaliert, dass der Mittelwert null und die Standardabweichung 1 ist. Diese Standardisierung wird erst nach Aufteilung in Trainings- und Testdatensätze durchgeführt, um Information Leakage zu verhindern.



```python
# Enthält Kundendaten von einer Kfz-Haftpflichtversicherung
freq = pd.read_csv("../data/freMTPL2freq.csv")

# Claim Anzahl entfernen (Erklärung siehe [4] Listing B.1)
freq = freq.drop(columns=["ClaimNb"])
freq["IDpol"] = freq["IDpol"].astype("int64")

```


```python
# Enthält die Schadenshöhe für jeden Schaden
sev = pd.read_csv("../data/freMTPL2sev.csv")

# Schadenshöhe und Vorkommen nach Kunden-ID aggregieren
sev_agg = sev
sev_agg["ClaimNb"] = 1
sev_agg = sev_agg.groupby("IDpol").sum()[["ClaimNb", "ClaimAmount"]].reset_index()
sev_agg = sev_agg.rename(columns={"ClaimAmount": "ClaimTotal"})

```


```python
# freq und sev zusammenführen --> Datensatz mit der korrekten Anzahl an Schadensmeldungen
freq = freq.merge(sev_agg, on="IDpol", how="left")
freq["ClaimNb"] = freq["ClaimNb"].fillna(0)
freq["ClaimTotal"] = freq["ClaimTotal"].fillna(0)

# Vehicle Brand als kategoriales Merkmal definieren um Reihenfolge der Brands festzulegen
freq["VehBrand"] = pd.Categorical(
    freq["VehBrand"],
    categories=[
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
        "B12",
        "B13",
        "B14",
    ],
)

# Area Codes einer Ordinalskala zuweisen (A=1, B=2,...)
freq["Area"] = pd.Categorical(freq["Area"], categories=["A", "B", "C", "D", "E", "F"])
freq["Area"] = freq["Area"].cat.codes + 1
freq = freq.rename(columns={"Area": "AreaCode"})

# Binäre Variable "VehGas" den Codes 0 und 1 zuordnen (Diesel = 0, Regular = 1)
freq["VehGas"] = pd.Categorical(freq["VehGas"], categories=["Diesel", "Regular"])
freq["VehGas"] = freq["VehGas"].cat.codes

# Datentyp von ClaimNb und Region anpassen
freq = freq.astype({"ClaimNb": "int64", "Region": "category"})

# Alle Einträge mit mehr als 5 Schadensmeldungen entfernen:
freq = freq[freq["ClaimNb"] <= 5]

# Exposure kann maximal 1 sein --> alle Beobachtungen mit höheren Werten auf 1 setzen:
freq["Exposure"] = freq["Exposure"].clip(lower=0, upper=1)

# Log(Density)
freq["log_Density"] = np.log(freq["Density"])
freq = freq.drop(columns=["Density"])

# Alle Einträge aus sev entfernen die jetzt nicht mehr in freq enthalten sind:
sev = sev[sev["IDpol"].isin(freq["IDpol"])][["IDpol", "ClaimAmount"]]

```


```python
freq.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IDpol</th>
      <th>Exposure</th>
      <th>AreaCode</th>
      <th>VehPower</th>
      <th>VehAge</th>
      <th>DrivAge</th>
      <th>BonusMalus</th>
      <th>VehBrand</th>
      <th>VehGas</th>
      <th>Region</th>
      <th>ClaimNb</th>
      <th>ClaimTotal</th>
      <th>log_Density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.10</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>55</td>
      <td>50</td>
      <td>B12</td>
      <td>1</td>
      <td>R82</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.104144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.77</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>55</td>
      <td>50</td>
      <td>B12</td>
      <td>1</td>
      <td>R82</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.104144</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.75</td>
      <td>2</td>
      <td>6</td>
      <td>2</td>
      <td>52</td>
      <td>50</td>
      <td>B12</td>
      <td>0</td>
      <td>R22</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.988984</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>0.09</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>46</td>
      <td>50</td>
      <td>B12</td>
      <td>0</td>
      <td>R72</td>
      <td>0</td>
      <td>0.0</td>
      <td>4.330733</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>0.84</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>46</td>
      <td>50</td>
      <td>B12</td>
      <td>0</td>
      <td>R72</td>
      <td>0</td>
      <td>0.0</td>
      <td>4.330733</td>
    </tr>
  </tbody>
</table>
</div>




```python
sev.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IDpol</th>
      <th>ClaimAmount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1552</td>
      <td>995.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1010996</td>
      <td>1128.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4024277</td>
      <td>1851.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4007252</td>
      <td>1204.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4046424</td>
      <td>1204.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Zufällige Störvariablen hinzufügen, um später für die Regression Attentions ein Maß für die Streung um 0 definieren zu können

# Normalverteilte Zufallsvariable RandN
freq["RandN"] = rng.standard_normal(size=(freq.shape[0], 1))

# Gleichverteilte Zufallsvariable RandU (standardisiert)
freq["RandU"] = rng.uniform(size=(freq.shape[0], 1))

```


```python
# Kategoriale Merkmale One-Hot Encoden (k-Kategorien führen zu k-1 Spalten)
categorical_columns = ["VehBrand", "Region"]
freq = pd.get_dummies(freq, columns=categorical_columns, drop_first=False)

```


```python
# Datensatz in Merkmale x und Zielvariable y aufteilen
y_freq = freq["ClaimNb"]
x_freq = freq.drop(columns=["IDpol", "ClaimNb", "ClaimTotal"])

```


```python
# Aufteilen in Trainings- und Testdaten
x_freq_train, x_freq_test, y_freq_train, y_freq_test = train_test_split(
    x_freq, y_freq, test_size=0.1
)

# Exposures getrennt speichern
exposures_train = x_freq_train["Exposure"]
exposures_test = x_freq_test["Exposure"]

x_freq_train = x_freq_train.drop(columns=["Exposure"])
x_freq_test = x_freq_test.drop(columns=["Exposure"])

```


```python
# Stetige und binäre Merkmale standardisieren:
continuous_columns = [
    "AreaCode",
    "BonusMalus",
    "log_Density",
    "DrivAge",
    "VehAge",
    "VehPower",
]
binary_columns = ["VehGas"]

x_freq_train_sc = x_freq_train.copy()
x_freq_test_sc = x_freq_test.copy()

# Trainings- und Testdatensatz werden getrennt standardisiert, um Information Leakage der Testdaten zu verhindern
scaler_freq = StandardScaler()
x_freq_train_sc[continuous_columns + binary_columns] = scaler_freq.fit_transform(
    x_freq_train_sc[continuous_columns + binary_columns]
)
x_freq_test_sc[continuous_columns + binary_columns] = scaler_freq.transform(
    x_freq_test_sc[continuous_columns + binary_columns]
)

# Zufallsvariable RandU standardisieren
scaler_freq_rand = StandardScaler()
x_freq_train_sc["RandU"] = scaler_freq_rand.fit_transform(x_freq_train_sc[["RandU"]])
x_freq_test_sc["RandU"] = scaler_freq_rand.transform(x_freq_test_sc[["RandU"]])

```

## 3.2 LocalGLMnet <a class="anchor" id="3_2-bullet"></a>

<font size="3">
Der Aufbau des LocalGLMnet ähnelt dem vorherigen bei dem synthetischen Datensatz. Da es sich jedoch um die Vorhersage von Zähldaten handelt, wird statt einer klassischen linearen Regression in der Response-Schicht in diesem Fall ein Poisson GLM verwendet. Die <em>Exposure</em> wird hierbei als Offset für das GLM eingesetzt. Durch One-Hot Encoding und hinzufügen der Variablen <em>RandN</em> und <em>RandU</em> ist die Inputdimension <em>q=45</em>.


```python
# LocalGLMnet Modell strukturieren

# LocalGLMnet nimmt als Input sowohl die Exposure als auch die Merkmale x
input_freq = tf.keras.Input(shape=(45), dtype="float32", name="Input")
vol_freq = tf.keras.Input(shape=(1), dtype="float32", name="Vol")

# Hidden Layer welche bis hin zur Attention Schicht mit 42 Neuronen (= Anzahl Inputmerkmale) führt
attention_freq = input_freq
attention_freq = tf.keras.layers.Dense(units=20, activation="tanh", name="Layer1")(
    attention_freq
)
attention_freq = tf.keras.layers.Dense(units=15, activation="tanh", name="Layer2")(
    attention_freq
)
attention_freq = tf.keras.layers.Dense(units=10, activation="tanh", name="Layer3")(
    attention_freq
)
attention_freq = tf.keras.layers.Dense(units=45, activation="linear", name="Attention")(
    attention_freq
)

# Skip-Connection
local_glm_freq = tf.keras.layers.Dot(name="LocalGLM", axes=1)(
    [input_freq, attention_freq]
)
# Fügt Intercept hinzu
local_glm_freq = tf.keras.layers.Dense(
    units=1, activation="exponential", name="Balance"
)(local_glm_freq)

# Response Schicht multipliziert Output des Netzes mit der Exposure
response_freq = tf.keras.layers.Multiply(name="Multiply")([local_glm_freq, vol_freq])

```


```python
# Modell kompilieren
local_glm_net_freq = tf.keras.Model(
    inputs=[input_freq, vol_freq], outputs=response_freq
)
local_glm_net_freq.compile(loss="poisson", optimizer="nadam")
local_glm_net_freq.summary()

```

    Model: "model_10"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    Input (InputLayer)              [(None, 45)]         0                                            
    __________________________________________________________________________________________________
    Layer1 (Dense)                  (None, 20)           920         Input[0][0]                      
    __________________________________________________________________________________________________
    Layer2 (Dense)                  (None, 15)           315         Layer1[0][0]                     
    __________________________________________________________________________________________________
    Layer3 (Dense)                  (None, 10)           160         Layer2[0][0]                     
    __________________________________________________________________________________________________
    Attention (Dense)               (None, 45)           495         Layer3[0][0]                     
    __________________________________________________________________________________________________
    LocalGLM (Dot)                  (None, 1)            0           Input[0][0]                      
                                                                     Attention[0][0]                  
    __________________________________________________________________________________________________
    Balance (Dense)                 (None, 1)            2           LocalGLM[0][0]                   
    __________________________________________________________________________________________________
    Vol (InputLayer)                [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Multiply (Multiply)             (None, 1)            0           Balance[0][0]                    
                                                                     Vol[0][0]                        
    ==================================================================================================
    Total params: 1,892
    Trainable params: 1,892
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
# Modell trainieren
history_freq = local_glm_net_freq.fit(
    [x_freq_train_sc, exposures_train],
    y_freq_train,
    batch_size=5000,
    epochs=100,
    validation_split=0.2,
)

```

    Epoch 1/100
    98/98 [==============================] - 2s 9ms/step - loss: 0.3123 - val_loss: 0.1945
    Epoch 2/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1795 - val_loss: 0.1676
    Epoch 3/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1678 - val_loss: 0.1626
    Epoch 4/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1642 - val_loss: 0.1601
    Epoch 5/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1620 - val_loss: 0.1584
    Epoch 6/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1605 - val_loss: 0.1573
    Epoch 7/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1593 - val_loss: 0.1563
    Epoch 8/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1585 - val_loss: 0.1557
    Epoch 9/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1578 - val_loss: 0.1553
    Epoch 10/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1574 - val_loss: 0.1550
    Epoch 11/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1571 - val_loss: 0.1548
    Epoch 12/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1568 - val_loss: 0.1546
    Epoch 13/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1566 - val_loss: 0.1545
    Epoch 14/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1565 - val_loss: 0.1544
    Epoch 15/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1563 - val_loss: 0.1543
    Epoch 16/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1562 - val_loss: 0.1542
    Epoch 17/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1561 - val_loss: 0.1542
    Epoch 18/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1560 - val_loss: 0.1541
    Epoch 19/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1559 - val_loss: 0.1540
    Epoch 20/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1558 - val_loss: 0.1539
    Epoch 21/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1558 - val_loss: 0.1539
    Epoch 22/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1557 - val_loss: 0.1538
    Epoch 23/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1556 - val_loss: 0.1538
    Epoch 24/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1556 - val_loss: 0.1538
    Epoch 25/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1555 - val_loss: 0.1538
    Epoch 26/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1555 - val_loss: 0.1537
    Epoch 27/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1554 - val_loss: 0.1537
    Epoch 28/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1553 - val_loss: 0.1537
    Epoch 29/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1553 - val_loss: 0.1536
    Epoch 30/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1553 - val_loss: 0.1536
    Epoch 31/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1552 - val_loss: 0.1537
    Epoch 32/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1552 - val_loss: 0.1536
    Epoch 33/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1551 - val_loss: 0.1536
    Epoch 34/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1551 - val_loss: 0.1536
    Epoch 35/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1551 - val_loss: 0.1536
    Epoch 36/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1550 - val_loss: 0.1536
    Epoch 37/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1550 - val_loss: 0.1537
    Epoch 38/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1550 - val_loss: 0.1536
    Epoch 39/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 40/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1549 - val_loss: 0.1536
    Epoch 41/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 42/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1548 - val_loss: 0.1536
    Epoch 43/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1548 - val_loss: 0.1536
    Epoch 44/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1548 - val_loss: 0.1536
    Epoch 45/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1548 - val_loss: 0.1536
    Epoch 46/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1548 - val_loss: 0.1536
    Epoch 47/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1535
    Epoch 48/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1547 - val_loss: 0.1536
    Epoch 49/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1547 - val_loss: 0.1537
    Epoch 50/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1547 - val_loss: 0.1536
    Epoch 51/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 52/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 53/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1546 - val_loss: 0.1537
    Epoch 54/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 55/100
    98/98 [==============================] - 1s 9ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 56/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 57/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1545 - val_loss: 0.1535
    Epoch 58/100
    98/98 [==============================] - 1s 9ms/step - loss: 0.1545 - val_loss: 0.1535
    Epoch 59/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1545 - val_loss: 0.1536
    Epoch 60/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1545 - val_loss: 0.1537
    Epoch 61/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1545 - val_loss: 0.1537
    Epoch 62/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1544 - val_loss: 0.1536
    Epoch 63/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1544 - val_loss: 0.1536
    Epoch 64/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1544 - val_loss: 0.1536
    Epoch 65/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1544 - val_loss: 0.1536
    Epoch 66/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1544 - val_loss: 0.1536
    Epoch 67/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1544 - val_loss: 0.1537
    Epoch 68/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1544 - val_loss: 0.1536
    Epoch 69/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1543 - val_loss: 0.1537
    Epoch 70/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1543 - val_loss: 0.1536
    Epoch 71/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1543 - val_loss: 0.1536
    Epoch 72/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1543 - val_loss: 0.1536
    Epoch 73/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1543 - val_loss: 0.1536
    Epoch 74/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1543 - val_loss: 0.1537
    Epoch 75/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1542 - val_loss: 0.1536
    Epoch 76/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1542 - val_loss: 0.1537
    Epoch 77/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1542 - val_loss: 0.1537
    Epoch 78/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1542 - val_loss: 0.1537
    Epoch 79/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1542 - val_loss: 0.1537
    Epoch 80/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1542 - val_loss: 0.1537
    Epoch 81/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1542 - val_loss: 0.1537
    Epoch 82/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1542 - val_loss: 0.1537
    Epoch 83/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1541 - val_loss: 0.1537
    Epoch 84/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1541 - val_loss: 0.1539
    Epoch 85/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1541 - val_loss: 0.1537
    Epoch 86/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1541 - val_loss: 0.1537
    Epoch 87/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1541 - val_loss: 0.1538
    Epoch 88/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1541 - val_loss: 0.1537
    Epoch 89/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1541 - val_loss: 0.1538
    Epoch 90/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1540 - val_loss: 0.1538
    Epoch 91/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1540 - val_loss: 0.1538
    Epoch 92/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1540 - val_loss: 0.1538
    Epoch 93/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1540 - val_loss: 0.1538
    Epoch 94/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1540 - val_loss: 0.1539
    Epoch 95/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1540 - val_loss: 0.1538
    Epoch 96/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1540 - val_loss: 0.1538
    Epoch 97/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1540 - val_loss: 0.1538
    Epoch 98/100
    98/98 [==============================] - 1s 6ms/step - loss: 0.1540 - val_loss: 0.1537
    Epoch 99/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1539 - val_loss: 0.1538
    Epoch 100/100
    98/98 [==============================] - 1s 5ms/step - loss: 0.1539 - val_loss: 0.1538


## 3.3 Auswertung <a class="anchor" id="3_3-bullet"></a>

<font size="3">
Im folgenden Teil soll zuerst eine Merkmalsselektion vorgenommen werden. Anschließend soll ein neues LocalGLMnet trainiert werden, welches nur Merkmale enthält, die eine Beziehung zur Zielvariable haben. Für diese werden daraufhin die Auswirkungen auf die Vorhersage sowie die Interaktionen zwischen Merkmalen betrachtet.

### 3.3.1 Variable Selection <a class="anchor" id="3_3_1-bullet"></a>

<font size="3">
Zuerst werden nun die Attention-Gewichte ermittelt. Auf Basis der Gewichte der Störvariablen <em>RandN</em> und <em>RandU</em> werden daraufhin Konfidenzintervalle bestimmt. Diese sind in Abbildung 5, der Darstellung der Attention-Gewichte in Abhängigkeit von den Ausgangsmerkmale, als blauer Bereich dargestellt. Hierdurch lässt sich bereits das Ausmaß des Zusammenhangs mit der Zielvariable abschätzen. Die finale Entscheidung, welche Merkmale entfernt werden, wird jedoch mithilfe der Coverage Ratio getroffen.


```python
# Über die Methode get_weights() erhält man die Kantengewichte, sowie den Bias für jeder Schicht
# --> man erhält also eine Liste mit numpy Arrays die in der Länge der Anzahl der Ebenen * 2 entspricht
weights_freq = local_glm_net_freq.get_weights()
for i in weights:
    print(i.shape, end=" | ")

```

    (8, 20) | (20,) | (20, 15) | (15,) | (15, 10) | (10,) | (10, 8) | (8,) | (1, 1) | (1,) | 


```python
# Neues Model ohne Response-Schicht --> ermöglicht auslesen der Attention Gewichte
# Benötigt als Input nur die Features, nicht die Exposures, da diese erst im späteren Layer erforderlich werden
weights_model_freq = tf.keras.Model(
    inputs=local_glm_net_freq.inputs[0],
    outputs=local_glm_net_freq.get_layer(name="Attention").output,
)

# Gewichte bestimmen
beta_x_freq = weights_model_freq.predict(x_freq_test_sc)

# Skalierung der Attention-Gewichte mithilfe des Gewichts der Response Schicht ( = Intercept beta_0)
beta_x_freq_scaled = beta_x_freq * local_glm_net_freq.get_weights()[8]

# Als DataFrame speichern um mittels der Merkmalsnamen auf die Attention Gewichte zugreifen zu können
beta_x_freq_scaled = pd.DataFrame(beta_x_freq_scaled, columns=x_freq_test_sc.columns)

```


```python
# Intervallgrenzen auf Basis der Zufallsvariablen RandN und RandU bestimmen

randn_mean = beta_x_freq_scaled["RandN"].mean()
randn_std = beta_x_freq_scaled["RandN"].std()

randu_mean = beta_x_freq_scaled["RandU"].mean()
randu_std = beta_x_freq_scaled["RandU"].std()

rand_mean = (randn_mean + randu_mean) / 2
rand_std = (randn_std + randu_std) / 2

print("Mittelwert RandN: " + str(randn_mean))
print("Standardabweichung RandN: " + str(randn_std))

print("\nMittelwert RandU: " + str(randu_mean))
print("Standardabweichung RandU: " + str(randu_std))

print("\nMittelwert Gesamt: " + str(rand_mean))
print("Standardabweichung Gesamt: " + str(rand_std))

# Intervalgrenzen bestimmen
alpha_freq = 0.001
bound_freq = stats.norm.ppf(alpha_freq / 2) * rand_std

print(
    "\nQuantil " + str(1 - alpha_freq / 2) + ": " + str(stats.norm.ppf(alpha_freq / 2))
)
print("Grenzen: ± " + str(abs(bound_freq)))

```

    Mittelwert RandN: 0.06710727
    Standardabweichung RandN: 0.08473704
    
    Mittelwert RandU: -0.040262144
    Standardabweichung RandU: 0.108501375
    
    Mittelwert Gesamt: 0.013422561809420586
    Standardabweichung Gesamt: 0.09661920368671417
    
    Quantil 0.9995: -3.2905267314918945
    Grenzen: ± 0.31792807250659316



```python
# Indizes der Testdaten zurücksetzen, damit sie mit beta_x_freq_scaled übereinstimmen
x_att = x_freq_test.copy()
x_att.reset_index(inplace=True)

```

<a class="anchor" id="abb5"></a>


```python
# Attention Plot freq-Datensatz

fig_freq_attention, axs_freq_att = plt.subplots(nrows=3, ncols=3, figsize=(30, 15))

# Merkmale festlegen für die ein Attention Subplot erstellt werden soll
columns = continuous_columns + binary_columns + ["RandN", "RandU"]

for i, ax in enumerate(axs_freq_att.flatten()):

    # Für VehGas wird ein Boxplot geplottet, da es eine binäre Variable ist und ein normaler Scatterplot nicht viel Sinn ergibt
    if columns[i] == "VehGas":
        diesel_index = x_att[x_att["VehGas"] == 0].index
        regular_index = x_att[x_att["VehGas"] == 1].index
        ax.boxplot(
            [
                beta_x_freq_scaled.loc[diesel_index]["VehGas"],
                beta_x_freq_scaled.loc[regular_index]["VehGas"],
            ],
            labels=["Diesel", "Regular"],
            zorder=10,
        )

    # Scatterplot für alle anderen Merkmale hinzufügen
    else:
        ax.scatter(
            x_att[columns[i]],
            beta_x_freq_scaled[columns[i]],
            s=0.5,
            c="black",
            zorder=10,
        )

    # x-Grenzen des Plots abfragen
    x_min, x_max = ax.get_xlim()

    # Intervallgrenzen
    interval = patches.Rectangle(
        xy=(x_min, -abs(bound_freq)),
        height=2 * abs(bound_freq),
        width=x_max - x_min,
        edgecolor="royalblue",
        facecolor="lightcyan",
        alpha=0.8,
        zorder=1,
    )
    ax.add_patch(interval)

    # Linien zur Verdeutlichung der Höhe der Attention Gewichte
    ax.hlines(y=0.25, xmin=x_min, xmax=x_max, colors="orange", linestyles="dashed")
    ax.hlines(y=-0.25, xmin=x_min, xmax=x_max, colors="orange", linestyles="dashed")
    ax.hlines(y=0, xmin=x_min, xmax=x_max, colors="red")

    # Layout
    ax.set_xlabel(columns[i])
    ax.set_ylabel("Regression Attention")
    ax.set_ylim((-1, 1))

fig_freq_attention.suptitle("Abbildung 5: Regression Attentions")
plt.show()

```


    
![png](/src/localGLMnet_markdown_assets/localGLMnet_82_0.png)
    


<font size="3">

Der Attention Plot lässt bereits darauf schließen, dass der Zusammenhang zwischen <em>VehGas</em>, <em>VehPower</em> und <em>AreaCode</em> mit der Zielvariable nur sehr gering ausfällt. Ein Großteil der Attention-Gewichte fällt in die durch die Störvariablen berechneten Konfidenzintervalle. Für <em>DriveAge</em>, <em>VehAge</em> und <em>BonusMalus</em> ist ein deutlicher Zusammenhang zu erkennen. Bevor auf diese mittels des Feature Contribution Plots genauer eingegangen wird, liefert der Hypothesentest auf Basis der Coverage Ratio eine Aussage darüber, welche Merkmale entfernt werden sollten:


```python
for col in columns:
    if col not in ["RandN", "RandU"]:
        size = beta_x_freq_scaled.shape[0]
        coverage = np.count_nonzero(
            beta_x_freq_scaled[col] < abs(bound_freq)
        ) - np.count_nonzero(beta_x_freq_scaled[col] < -abs(bound_freq))
        coverage_ratio = coverage / size
        print("Coverage Ratio " + col + ": " + str(coverage_ratio))

```

    Coverage Ratio AreaCode: 0.9636436040766361
    Coverage Ratio BonusMalus: 0.2912198935118951
    Coverage Ratio log_Density: 0.9910620787304022
    Coverage Ratio DrivAge: 0.5153463813218094
    Coverage Ratio VehAge: 0.8742643913806581
    Coverage Ratio VehPower: 0.9967404610551467
    Coverage Ratio VehGas: 0.08660639223610271


<font size="3">
Generell weichen die Coverage Ratios leicht von denen im Paper ab. Dies kann mit einer unterschiedlichen Initialisierung der Zufallsgeneratoren zusammenhängen.
Anhand der Coverage Ratios kann lediglich <em>VehPower</em> entfernt werden, da die Coverage Ratio über 0.999 liegt. Ihr Einfluss auf die Vorhersage entspricht also in etwa dem einer zufälligen Störvariable. Da <em>AreaCode</em> und <em>Density</em> mit 99.37% und 98.86 % nur knapp unter 99,9% liegen, wird im Bezug auf die empirische Analyse von Noll et al. auf die Korrelation der beiden Variablen verwiesen, welche auch in Abbildung 6 zu erkennen ist [1, 5]. Da eine der beiden Variablen ausreicht, um alle Informationen zu erhalten wird <em>AreaCode</em> entfernt. <br>

<a class="anchor" id="abb6"></a>


```python
area_density = []
labels = []

for i in np.sort(x_att["AreaCode"].unique()):
    index = x_att[x_att["AreaCode"] == i].index
    area_density.append(x_att.loc[index]["log_Density"])
    labels.append(int(i))

plt.boxplot(x=area_density, labels=labels)
plt.xlabel("Area Code")
plt.ylabel("log(Density)")
plt.title("Abbildung 6: Area Code vs. Density")
plt.show()

```


    
![png](/src/localGLMnet_markdown_assets/localGLMnet_87_0.png)
    


### 3.3.2 Neues LocalGLMnet trainieren <a class="anchor" id="3_3_2-bullet"></a>

<font size="3">
Nachdem nun <em>AreaCode</em>, <em>VehPower</em> sowie die Zufallskomponenten <em>RandN</em> und <em>RandU</em> entfernt wurden, kann ein neues Modell mit reduzierter Inputdimension trainiert werden. Insgesamt enthält der Datensatz also noch 41 Merkmale.


```python
# Nur Merkmale behalten die signifikant sind (RandN, RandU, AreaCode, VehPower entfernen)
x_freq_sig_train = x_freq_train.drop(columns=["RandN", "RandU", "AreaCode", "VehPower"])
x_freq_sig_test = x_freq_test.drop(columns=["RandN", "RandU", "AreaCode", "VehPower"])

sig_columns = continuous_columns + binary_columns
sig_columns.remove("VehPower")
sig_columns.remove("AreaCode")
print(sig_columns)

```

    ['BonusMalus', 'log_Density', 'DrivAge', 'VehAge', 'VehGas']



```python
# Neuen Datensatz mit wichtigen Spalten skalieren (mean=0, std=1)
x_freq_sig_train_sc = x_freq_sig_train.copy()
x_freq_sig_test_sc = x_freq_sig_test.copy()

scaler_freq_sig = StandardScaler()
x_freq_sig_train_sc[sig_columns] = scaler_freq.fit_transform(
    x_freq_sig_train_sc[sig_columns]
)
x_freq_sig_test_sc[sig_columns] = scaler_freq.transform(x_freq_sig_test_sc[sig_columns])

```


```python
# Neues LocalGLMnet erstellen (41 Input Merkmale)
input_freq_sig = tf.keras.Input(shape=(41), dtype="float32", name="Input")
vol_freq_sig = tf.keras.Input(shape=(1), dtype="float32", name="Vol")

attention_freq_sig = input_freq_sig
attention_freq_sig = tf.keras.layers.Dense(units=20, activation="tanh", name="Layer1")(
    attention_freq_sig
)
attention_freq_sig = tf.keras.layers.Dense(units=15, activation="tanh", name="Layer2")(
    attention_freq_sig
)
attention_freq_sig = tf.keras.layers.Dense(units=10, activation="tanh", name="Layer3")(
    attention_freq_sig
)
attention_freq_sig = tf.keras.layers.Dense(
    units=41, activation="linear", name="Attention"
)(attention_freq_sig)

# Skip-Connection
local_glm_freq_sig = tf.keras.layers.Dot(name="LocalGLM", axes=1)(
    [input_freq_sig, attention_freq_sig]
)
# Fügt Intercept hinzu
local_glm_freq_sig = tf.keras.layers.Dense(
    units=1, activation="exponential", name="Balance"
)(local_glm_freq_sig)

# Response Schicht
response_freq_sig = tf.keras.layers.Multiply(name="Multiply")(
    [local_glm_freq_sig, vol_freq_sig]
)

```


```python
# Modell kompilieren
local_glm_net_freq_sig = tf.keras.Model(
    inputs=[input_freq_sig, vol_freq_sig], outputs=response_freq_sig
)
local_glm_net_freq_sig.compile(loss="poisson", optimizer="nadam")
local_glm_net_freq_sig.summary()

```

    Model: "model_12"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    Input (InputLayer)              [(None, 41)]         0                                            
    __________________________________________________________________________________________________
    Layer1 (Dense)                  (None, 20)           840         Input[0][0]                      
    __________________________________________________________________________________________________
    Layer2 (Dense)                  (None, 15)           315         Layer1[0][0]                     
    __________________________________________________________________________________________________
    Layer3 (Dense)                  (None, 10)           160         Layer2[0][0]                     
    __________________________________________________________________________________________________
    Attention (Dense)               (None, 41)           451         Layer3[0][0]                     
    __________________________________________________________________________________________________
    LocalGLM (Dot)                  (None, 1)            0           Input[0][0]                      
                                                                     Attention[0][0]                  
    __________________________________________________________________________________________________
    Balance (Dense)                 (None, 1)            2           LocalGLM[0][0]                   
    __________________________________________________________________________________________________
    Vol (InputLayer)                [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Multiply (Multiply)             (None, 1)            0           Balance[0][0]                    
                                                                     Vol[0][0]                        
    ==================================================================================================
    Total params: 1,768
    Trainable params: 1,768
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
# Modell trainieren
history_freq_sig = local_glm_net_freq_sig.fit(
    [x_freq_sig_train_sc, exposures_train],
    y_freq_train,
    batch_size=5000,
    epochs=100,
    validation_split=0.2,
)

```

    Epoch 1/100
    98/98 [==============================] - 2s 9ms/step - loss: 0.3150 - val_loss: 0.1937
    Epoch 2/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1776 - val_loss: 0.1646
    Epoch 3/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1646 - val_loss: 0.1591
    Epoch 4/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1610 - val_loss: 0.1569
    Epoch 5/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1593 - val_loss: 0.1558
    Epoch 6/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1583 - val_loss: 0.1552
    Epoch 7/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1577 - val_loss: 0.1548
    Epoch 8/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1573 - val_loss: 0.1546
    Epoch 9/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1571 - val_loss: 0.1544
    Epoch 10/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1569 - val_loss: 0.1543
    Epoch 11/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1567 - val_loss: 0.1543
    Epoch 12/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1566 - val_loss: 0.1541
    Epoch 13/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1565 - val_loss: 0.1541
    Epoch 14/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1564 - val_loss: 0.1540
    Epoch 15/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1563 - val_loss: 0.1540
    Epoch 16/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1562 - val_loss: 0.1539
    Epoch 17/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1561 - val_loss: 0.1539
    Epoch 18/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1561 - val_loss: 0.1539
    Epoch 19/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1560 - val_loss: 0.1538
    Epoch 20/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1560 - val_loss: 0.1538
    Epoch 21/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1559 - val_loss: 0.1538
    Epoch 22/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1559 - val_loss: 0.1537
    Epoch 23/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1558 - val_loss: 0.1537
    Epoch 24/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1558 - val_loss: 0.1537
    Epoch 25/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1557 - val_loss: 0.1537
    Epoch 26/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1557 - val_loss: 0.1536
    Epoch 27/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1556 - val_loss: 0.1536
    Epoch 28/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1556 - val_loss: 0.1536
    Epoch 29/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1556 - val_loss: 0.1536
    Epoch 30/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1555 - val_loss: 0.1536
    Epoch 31/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1555 - val_loss: 0.1536
    Epoch 32/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1555 - val_loss: 0.1536
    Epoch 33/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1555 - val_loss: 0.1535
    Epoch 34/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1554 - val_loss: 0.1536
    Epoch 35/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1554 - val_loss: 0.1535
    Epoch 36/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1554 - val_loss: 0.1535
    Epoch 37/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1554 - val_loss: 0.1535
    Epoch 38/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1553 - val_loss: 0.1535
    Epoch 39/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1553 - val_loss: 0.1535
    Epoch 40/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1553 - val_loss: 0.1534
    Epoch 41/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1553 - val_loss: 0.1535
    Epoch 42/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1552 - val_loss: 0.1535
    Epoch 43/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1552 - val_loss: 0.1535
    Epoch 44/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1552 - val_loss: 0.1535
    Epoch 45/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1552 - val_loss: 0.1535
    Epoch 46/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1552 - val_loss: 0.1535
    Epoch 47/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1552 - val_loss: 0.1534
    Epoch 48/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1551 - val_loss: 0.1535
    Epoch 49/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1551 - val_loss: 0.1535
    Epoch 50/100
    98/98 [==============================] - 1s 9ms/step - loss: 0.1551 - val_loss: 0.1535
    Epoch 51/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1551 - val_loss: 0.1534
    Epoch 52/100
    98/98 [==============================] - 1s 9ms/step - loss: 0.1551 - val_loss: 0.1535
    Epoch 53/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1551 - val_loss: 0.1535
    Epoch 54/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1550 - val_loss: 0.1535
    Epoch 55/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1550 - val_loss: 0.1534
    Epoch 56/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1550 - val_loss: 0.1535
    Epoch 57/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1550 - val_loss: 0.1534
    Epoch 58/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1550 - val_loss: 0.1534
    Epoch 59/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1550 - val_loss: 0.1534
    Epoch 60/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1549 - val_loss: 0.1536
    Epoch 61/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 62/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 63/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 64/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 65/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 66/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 67/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 68/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1549 - val_loss: 0.1535
    Epoch 69/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1548 - val_loss: 0.1536
    Epoch 70/100
    98/98 [==============================] - 1s 10ms/step - loss: 0.1548 - val_loss: 0.1535
    Epoch 71/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1548 - val_loss: 0.1535
    Epoch 72/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1548 - val_loss: 0.1535
    Epoch 73/100
    98/98 [==============================] - 1s 9ms/step - loss: 0.1548 - val_loss: 0.1535
    Epoch 74/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1548 - val_loss: 0.1536
    Epoch 75/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1548 - val_loss: 0.1535
    Epoch 76/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1536
    Epoch 77/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1548 - val_loss: 0.1535
    Epoch 78/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1536
    Epoch 79/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1535
    Epoch 80/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1535
    Epoch 81/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1535
    Epoch 82/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1535
    Epoch 83/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1535
    Epoch 84/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1547 - val_loss: 0.1537
    Epoch 85/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1536
    Epoch 86/100
    98/98 [==============================] - 1s 9ms/step - loss: 0.1547 - val_loss: 0.1535
    Epoch 87/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1547 - val_loss: 0.1536
    Epoch 88/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 89/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 90/100
    98/98 [==============================] - 1s 9ms/step - loss: 0.1546 - val_loss: 0.1537
    Epoch 91/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 92/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1546 - val_loss: 0.1535
    Epoch 93/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1546 - val_loss: 0.1535
    Epoch 94/100
    98/98 [==============================] - 1s 12ms/step - loss: 0.1546 - val_loss: 0.1537
    Epoch 95/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 96/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 97/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1546 - val_loss: 0.1536
    Epoch 98/100
    98/98 [==============================] - 1s 8ms/step - loss: 0.1546 - val_loss: 0.1535
    Epoch 99/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1545 - val_loss: 0.1536
    Epoch 100/100
    98/98 [==============================] - 1s 7ms/step - loss: 0.1545 - val_loss: 0.1536


### 3.3.3 Feature Contribution <a class="anchor" id="3_3_3-bullet"></a>

<font size="3">
Der Feature Contribution Plot stellt für die noch vorhandenen (kontinuierlichen und binären) Merkmale den Einfluss der Merkmale dar. Die Feature Contribution ergibt sich wie bereits beim synthetischen Datensatz durch das Produkt aus Attention-Gewicht und (skaliertem) Inputmerkmal. Zuerst werden hierfür daher die Attention-Gewichte extrahiert.


```python
# Neues Model ohne Response-Schicht --> ermöglicht auslesen der Attention Gewichte
# benötigt als Input nur die Features, nicht die Exposures, da diese erst im späteren Layer erforderlich werden
weights_model_freq_sig = tf.keras.Model(
    inputs=local_glm_net_freq_sig.inputs[0],
    outputs=local_glm_net_freq_sig.get_layer(name="Attention").output,
)

# Gewichte bestimmen
beta_x_freq_sig = weights_model_freq_sig.predict(x_freq_sig_test_sc)

# Skalierung der Attention-Gewichte mithilfe des Gewichts der Response Schicht ( = Intercept beta_0)
beta_x_freq_sig_sc = beta_x_freq_sig * local_glm_net_freq_sig.get_weights()[8]

# Als DataFrame speichern um mittels der Merkmalsnamen auf die Attention Gewichte zugreifen zu können
beta_x_freq_sig_sc = pd.DataFrame(
    beta_x_freq_sig_sc, columns=x_freq_sig_test_sc.columns
)

```

<a class="anchor" id="abb7"></a>


```python
# Feature Contribution Plot freq-Datensatz

# Index der Testdaten zurücsetzen
x_con = x_freq_sig_test.reset_index(drop=True)
# Feature Contribution berechnen (mit skalierten Featurewert)
feature_con = x_freq_sig_test_sc.reset_index(drop=True) * beta_x_freq_sig_sc

fig_freq_con = plt.figure(tight_layout=True, figsize=(30, 15))

spec = GridSpec(ncols=6, nrows=2, figure=fig_freq_con)
ax1_freq_con = fig_freq_con.add_subplot(spec[0, 0:2])
ax2_freq_con = fig_freq_con.add_subplot(spec[0, 2:4])
ax3_freq_con = fig_freq_con.add_subplot(spec[0, 4:6])
ax4_freq_con = fig_freq_con.add_subplot(spec[1, 1:3])
ax5_freq_con = fig_freq_con.add_subplot(spec[1, 3:5])

axs_freq_con = [ax1_freq_con, ax2_freq_con, ax3_freq_con, ax4_freq_con, ax5_freq_con]

for i in range(len(axs_freq_con)):

    # Feature Contribution Splines berechnen
    # Feature Contribution = beta(xi)*xi

    # Wenn es sich um das Merkmal "VehGas" handelt, wird ein Boxplot hinzugefügt
    if sig_columns[i] == "VehGas":
        diesel_index = x_con[x_con["VehGas"] == 0].index
        regular_index = x_con[x_con["VehGas"] == 1].index
        axs_freq_con[i].boxplot(
            [
                feature_con.loc[diesel_index]["VehGas"],
                feature_con.loc[regular_index]["VehGas"],
            ],
            labels=["Diesel", "Regular"],
            zorder=10,
        )
        x_min, x_max = axs_freq_con[i].get_xlim()

    # Ansonsten wird ein Scatterplot + Spline hinzugefügt
    else:

        contribution = np.column_stack(
            [x_con[sig_columns[i]], feature_con[sig_columns[i]]]
        )
        con_ind = np.lexsort((contribution[:, 1], contribution[:, 0]))
        contribution_sorted = contribution[con_ind]

        con_spline = interpolate.UnivariateSpline(
            contribution_sorted[:, 0], contribution_sorted[:, 1]
        )

        # Scatter Plot --> x: Werte der Inputfeatures, y:Feature Contribution (β(x)*x)
        axs_freq_con[i].scatter(
            contribution[:, 0], contribution[:, 1], s=0.5, zorder=10
        )

        # X min und x max festlegen (min = kleinster Wert, max= mean+3*std)
        x_min = x_con[sig_columns[i]].min()
        x_max = x_con[sig_columns[i]].mean() + 3 * x_con[sig_columns[i]].std()
        xs = np.linspace(x_min, x_max, 1000)

        # Feature Contribution Spline plotten
        axs_freq_con[i].plot(xs, con_spline(xs), color="purple", zorder=20)
        axs_freq_con[i].set_xlim((x_min, x_max))

    # Hinzufügen von horizontalen Linien um die Stärke der Feature Contribution zu visualisieren
    axs_freq_con[i].hlines(
        y=0, xmin=x_min, xmax=x_max, colors="red", alpha=0.7, zorder=1
    )
    axs_freq_con[i].hlines(
        y=0.25, xmin=x_min, xmax=x_max, colors="orange", linestyles="dashed"
    )
    axs_freq_con[i].hlines(
        y=-0.25, xmin=x_min, xmax=x_max, colors="orange", linestyles="dashed"
    )

    # Layout
    axs_freq_con[i].set_ylim((-2, 2))
    axs_freq_con[i].set_xlabel(sig_columns[i])
    axs_freq_con[i].set_title("Feature Contribution: " + sig_columns[i])
    axs_freq_con[i].set_ylabel("Feature Contribution")

fig_freq_con.suptitle("Abbildung 7: Feature Contribution")
plt.show()

```


    
![png](/src/localGLMnet_markdown_assets/localGLMnet_99_0.png)
    


<font size="3">
Für jedes der verbleibenden Inputmerkmale ist ein deutlicher funktionaler Zusammenhang zur Zielvariable zu erkennen. Dieser wird zur Veranschaulichung durch einen Spline approximiert.<br> Vor allem bei <em>BonusMalus</em>, <em>DrivAge</em> und <em>VehAge</em> zeigen sich nachvollziehbare Zusammenhänge. Bei dem Fahreralter ist die Feature Contribution bspw. besonders für sehr junge Fahrer und sehr alte Fahrer hoch. Zu Beginn kann sich dies auf fehlende Erfahrung, später auf nachlassende Reaktionsfähigkeit oder Sehstärke zurückführen. Die Feature Contribution des Bonus-Malus verläuft in etwa gespiegelt, weshalb Richmann und Wüthrich eine Interaktion zwischen den beiden Variablen unterstellen [1]. Dies liegt daran, dass ein Fahranfänger im Bonus-Malus-System bei 100 startet und dieser Wert erst mit zunehmender Erfahrung abnimmt. Dieser Zusammenhang lässt sich ebenfalls in Abbildung 6 des Papers erkennen [1].

#### __Feature Contribution kategorialer Merkmale__

<font size="3">
Um die Feature Contribution von kategorialen Merkmalen darzustellen, müssen diese mittels One-Hot-Encoding und nicht Dummy-Encoding (k-Merkmale führen zu k-1 Spalten) encodiert worden sein. Bei Dummy Encoding wäre es nicht möglich, die Feature Contribution für die wegfallende Kategorie zu berechnen, da sie keiner Spalte zugeordnet werden kann. Eine genaue Erläuterung der Vor- und Nachteile von One-Hot Encoding gegenüber Dummy Encoding findet sich in Richman/Wüthrich Abschnitt 3.6 [1].<br>
Da es sich bei den Werten der One-Hot encodierten Kategorien nur um 0 oder 1 handelt, entsprechen die Attention-Gewichte ebenfalls der Feature Contribution (⁠β*x). Sie werden in Abbildung 8 dargestellt. Während die Darstellung für relativ wenige Ausprägungen möglich ist, wird sie schnell unübersichtlich.


```python
regions_con = beta_x_freq_sig_sc.filter(regex="Region*")
regions_con.columns = regions_con.columns.str.replace("Region_", "")

brands_con = beta_x_freq_sig_sc.filter(regex="VehBrand*")
brands_con.columns = brands_con.columns.str.replace("VehBrand_", "")

```

<a class="anchor" id="abb8"></a>


```python
fig_cat, axs_cat = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

axs_cat[0].boxplot(x=regions_con)
axs_cat[0].set_xticklabels(labels=regions_con.columns, rotation=90)
axs_cat[0].set_ylabel("Feature Contribution")
axs_cat[0].set_xlabel("Regions")
axs_cat[0].set_title("Feature Contribution: Regions")

axs_cat[1].boxplot(x=brands_con)
axs_cat[1].set_xticklabels(labels=brands_con.columns, rotation=90)
axs_cat[1].set_ylabel("Feature Contribution")
axs_cat[1].set_xlabel("Vehicle Brand")
axs_cat[1].set_title("Feature Contribution: VehBrand")

plt.suptitle("Abbildung 8: Feature Contribution kategorialer Variablen")

plt.show()

```


    
![png](/src/localGLMnet_markdown_assets/localGLMnet_105_0.png)
    


<font size="3">
Jeder Boxplot entspricht einer Merkmalsausprägung der Kategorie <em>Region</em> bzw. <em>VehBrand</em>. Der Median veranschaulicht die Höhe der Feature Contribution. Je größer die Box ist, umso mehr Interaktionen mit anderen Merkmalen gibt es.<br>
Die stärkste Auswirkung auf die Vorhersage bei den Regionen hat die Ausprägung <em>R25</em>. Bei den Fahrzeugmarken fällt vor allem <em>B9</em> auf.

### 3.3.4 Interaction Strengths <a class="anchor" id="3_3_4-bullet"></a>

<font size="3">
Um die Interaktionen der (binären und stetigen) Merkmale zu analysieren, werden erneut die Gradienten mithilfe von Splines dargestellt. Hierfür werden zuerst wie auch bei dem synthetischen Datensatz die Gradienten ermittelt.


```python
# Stetige Spalten festlegen (Splines nur für stetige Spalten)

sig_continuous_columns = ["BonusMalus", "log_Density", "DrivAge", "VehAge"]
sig_continuous_columns_id = []
sig_columns_id = []

# Spaltenindex der stetigen Merkmale ermitteln
for col in sig_continuous_columns:
    sig_continuous_columns_id.append(x_freq_sig_train.columns.get_loc(col))

# Spaltenindex der signifikanten Merkmale ermitteln (stetige Merkmale + VehGas)
for col in sig_columns:
    sig_columns_id.append(x_freq_sig_train.columns.get_loc(col))

```


```python
# Gradienten bestimmen

gradients_freq = []
x_freq_grad = tf.constant(x_freq_sig_train_sc)

# Für jede Inputvariable wird ein Modell gefittet, um anschließend die partiellen Ableitungen auslesen zu können
for i in range(len(sig_continuous_columns)):

    # Lambda Layer als Output Schicht, um beta_i als Output zu erhalten (partielle Ableitungen ∂β_j(x)/∂x_j')
    beta = attention_freq_sig
    beta = tf.keras.layers.Lambda(lambda x: x[:, sig_continuous_columns_id[i]])(beta)
    grad_model = tf.keras.Model(inputs=input_freq_sig, outputs=beta)

    # GradientTape ermöglicht das auslesen der Gradienten
    with tf.GradientTape() as g:
        g.watch(x_freq_grad)
        pred_attention = grad_model.call(x_freq_grad)

    grad = g.gradient(pred_attention, x_freq_grad)

    # Array das sowohl den Wert von x, als auch den entsprechenden Wert von βk(x) enthält
    grad_wrt_x = np.column_stack(
        (x_freq_sig_train[sig_continuous_columns[i]], grad.numpy())
    )

    # Um später die Splines zu modellieren muss die x-Komponente monoton steigend sein --> sortieren des Arrays
    ind = np.lexsort((grad_wrt_x[:, 2], grad_wrt_x[:, 0]))
    grad_wrt_x_sorted = grad_wrt_x[ind]

    # Gradienten in Liste speichern
    gradients_freq.append(grad_wrt_x_sorted)

```


```python
# Univariate Splines modellieren, um die Interaktion zwischen Features darzustellen
freq_splines = []

# Für alle Attention Gewichte β (nur stetige Merkmale)
for i in range(len(sig_continuous_columns)):
    freq_splines.append([])

    # Für alle Inputvariablen x (auch VehGas)
    for j in range(len(sig_columns_id)):
        freq_splines[i].append(
            interpolate.UnivariateSpline(
                gradients_freq[i][:, 0],
                gradients_freq[i][:, sig_columns_id[j] + 1],
            )
        )

```

<a class="anchor" id="abb9"></a>


```python
# Spline Interaction Plot freq-Datensatz

# x_freq_spline enthält nur stetige Spalten
x_freq_spline = x_freq_sig_train[sig_continuous_columns]

fig_freq_spline, axs_freq_spline = plt.subplots(nrows=2, ncols=2, figsize=(30, 15))

for i, ax in enumerate(axs_freq_spline.flatten()):

    # x_min und x_max festlegen und darauf basierend Linspace für die Splines erzeugen
    x_min = x_freq_spline.iloc[:, i].min()
    x_max = x_freq_spline.iloc[:, i].mean() + 3 * x_freq_spline.iloc[:, i].std()

    xs = np.linspace(x_min, x_max, 1000)

    # Spline für alle noch vorhandenen Merkmale
    for j in range(len(sig_columns)):
        ax.plot(xs, freq_splines[i][j](xs), label=sig_columns[j])

    # Hinzufügen von horizontalen Linien um die Stärke der Feature Contribution zu visualisieren
    ax.hlines(y=0.25, xmin=x_min, xmax=x_max, colors="black", linestyles="dashed")
    ax.hlines(y=-0.25, xmin=x_min, xmax=x_max, colors="black", linestyles="dashed")
    ax.hlines(y=0, xmin=x_min, xmax=x_max, colors="black")

    # Inline Labels und Legende hinzufügen
    labelLines(ax.get_lines(), zorder=2.5)
    ax.legend(loc="lower right", ncol=2)

    # Layout
    ax.set_xlabel(sig_continuous_columns[i])
    ax.set_ylabel("Interaction Strengths")
    ax.set_ylim((-2, 2))
    ax.set_title("Interactions of feature component: " + sig_continuous_columns[i])


fig_freq_spline.suptitle("Abbildung 9: Interaction Strengths")
plt.show()

```


    
![png](/src/localGLMnet_markdown_assets/localGLMnet_113_0.png)
    


<font size="3">
Durch die Splines lassen sich die Interaktionen der Merkmale analysieren. <em>Density</em> und <em>VehAge</em> weisen fast keine Interaktionen mit anderen Merkmalen auf und verhalten sich linear. <em>DrivAge</em> und <em>BonusMalus</em> sind hingegen nichtlinear und interagieren miteinander.

<hr>

# 4. Zusammenfassung <a class="anchor" id="4-bullet"></a>

<hr>

<font size="3">
Zusammenfassend lässt sich sagen, dass der LocalGLMnet Ansatz vor allem in Bezug auf die Erklärbarkeit und somit auch Merkmalsselektion deutliche Vorteile gegenüber klassischen FFNs bietet. Anstatt Erklärbarkeit nur nachträglich durch Ansätze wie Surrogatmodelle oder Partial Dependency Plots zu erzeugen, erlaubt das LocalGLMnet bereits durch seine Struktur bereits ein gewisses Maß an Erklärbarkeit. Ein Vergleich von diesen klassischen Ansätzen zum LocalGLMnet findet sich ebenfalls im ursprünglichen Paper und dessen Anhang. <br>
Richmann & Wüthrich können sich aufgrund der guten Erklärbarkeit bei gleichzeitig hoher Vorhersagegenauigkeit unterschiedliche Anwendungszwecke vorstellen. Es kann sowohl als eigenständiges Netz, aber auch als Surrogatmodell oder als Vorläufer eines klassischen FFNs zur initialen Merkmalselektion verwendet werden [1].<br>
Vor allem für Aktuare, welche den Umgang mit GLMs gewohnt sind, bietet das LocalGLMnet einen interessanten Ansatz bei dem die Erklärbarkeit nicht vollständig für die Vorhersagegenauigkeit aufgegeben werden muss. Ein einziger Nachteil ist, dass das Modell zur Zeit hauptsächlich für strukturierte, möglichst stetige oder binäre Daten optimiert ist. Um weitere Anwendungsfelder zu erschließen, benötigt es weiterer Forschung.

# Literaturverzeichnis <a class="anchor" id="literatur"></a>

<font size="3">
[1] Ronald Richman und Mario V. Wüthrich. 2022. LocalGLMnet: interpretable deep learning for tabular data. Scandinavian Actuarial Journal 2022, 1, 71–95. DOI: <a href="https://doi.org/10.1080/03461238.2022.2081816">https://doi.org/10.1080/03461238.2022.2081816</a> <br>
<br>
[2] John A. Nelder und Robert W. M. Wedderburn. 1972. Generalized Linear Models. Journal of the Royal Statistical Society, Vol. 135, No. 3, 370–384 DOI: <a href="https://doi.org/10.2307/2344614">https://doi.org/10.2307/2344614</a>.<br>
<br>
[3] Martin Seehafer, Stefan Nörtemann, Jonas Offtermatt, Fabian Transchel, Axel Kiermaier, René Külheim, und Wiltrud Weidner. 2021. Actuarial Data Science. De Gruyter.<br>
<br>
[4] Mario V. Wüthrich und Michael Merz. 2023. Statistical Foundations of Actuarial Learning and its Applications. Springer International Publishing, Cham.<br>
<br>
[5] Alexander Noll, Robert Salzmann, und Mario V.Wuthrich. 2020. Case Study: French Motor Third-Party Liability Claims <a href="http://dx.doi.org/10.2139/ssrn.3164764">http://dx.doi.org/10.2139/ssrn.3164764</a>
