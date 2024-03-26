# Machine Perception and Tracking - Praxisprojekt 1
## Modus Operandi
Sie wurden in Gruppen zu je maximal drei Studierenden eingeteilt.
Organisieren Sie ihre Arbeit selbst. Die Entwicklung soll hauptsächlich
während der vorgesehenen Praktikumszeit erfolgen. Der Ihnen so 
zur Verfügung stehende Zeitumfang reicht aus um die Aufgabe im Team zu bewältigen. Es ist jedoch wichtig das sie kontinuierlich an diesem Projekt arbeiten und die Arbeiten nicht vor sich herschieben. Motivieren und unterstützen Sie sich gegenseitig.

Forken Sie dieses Repository und sorgen Sie dafür das ihre Teamkollegen
ebenfalls am Code mitarbeiten können. 

    Ich möchte später von jedem
    ihrer Kollegen eigene commits sehen!


## Abgabe und Vorstellung
Die Teams arbeiten für sich und stellen ihre Ergebnisse unabhängig voneinander vor. Jedes Team hat für seine Vorstellung 10 Minuten Zeit.
Ziel ist in diesen 10 Minuten die komplette Pipeline mit allen Funktionalitäten zu zeigen. 

Ihre Pipeline sollte soweit vorbereitet sein das Sie Aufnahmen von den Gesichtern ihrer Teammitglieder haben. Zusätzlich sollen Sie während ihres Vortrages einen ihrer Kommilitonen, der nicht in ihrem Team ist, Aufnehmen und mit eintrainieren. Am Ende ihres 10-minütigen Vortrages soll ihre Pipeline in der Lage sein ihre Teamkollegen sowie den 
neuen Kommilitonen zu erkennen. 

## Bonus für alle Teams:
Organisieren Sie sich über ihre Teamgrenzen hinweg und sammeln
Sie möglichst viele Bilder von den Mitstudierenden aus ihrem Kurs. 
Trainieren Sie eine großes Modell welches all diese Gesichter gleichzeitig unterscheidet und führen Sie dieses ebenfalls live vor. 

Für diese Aufgabe genügt es wenn ein Team ihre jeweilige Codebase verwendet.

## Bewertungskriterien
**Funktionalität und Korrektheit**. Ihr Code muß funktionieren und den formalen Anforderungen genügen. Das bedeutet insbesondere das die weiter unten beschrieben Modi ihres Programms wie dargestellt funktionieren müssen. Sie weisen dies durch ihre Live-Vorführung nach.

**Modularisierung und Linting**: Ihr Code soll modular und aufgeräumt sein. Wenn sie Code-Teile an mehreren Stellen wiederverwenden vermeiden Sie es diesen Code zu duplizieren. Lassen Sie einen linter 
über ihren Code laufen, z.B. [BLACK](https://github.com/psf/black). 
Ich werde das nach Ihrer Abgabe prüfen.

**Eigenleistung**: Ihr Code sollte selbst von Ihnen stammen und nicht
von anderen Teams kopiert werden. Im Zweifel müssen Sie erklären können
was ihr code tut und warum Sie bestimmte Lösungen gewählt haben. Jdes Team-Mitglied soll einen eigenen Beitrag geleistet haben. Dies sollte
anhand von eigenen Code-Commits im Repository auch sichtbar sein!

    Noch ein Hinweis: Da ich am Ende alle Projekte sehen werde, werde ich auch sehen wenn Code-Abschnitt kopiert oder geteilt wurden. Da ich dann im Zweifel nicht feststellen kann wer von wem abgeschrieben hat muß ich dies in die Bewertung beider Teams einfließen lassen! 

## Face Recognition Pipeline
Die Aufgabe in diesem Praxisprojekt besteht darin eine vollständige
Datenpipeline für die Erkennung von Personen in Kamerabildern zu entwickeln. Dabei soll ein Pythonskript geschrieben werden welches über die Kommandozeile steuerbar ist und dabei mehrere Funktionalitäten abdeckt. Das Skript soll dabei die folgenden Funktionen unterstützen:

### Recording
Mit dem Aufruf

    python main.py record --folder dennis

soll ein Live-Bild der angeschlossenen Webcam aufgerufen werden.

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

Der in OpenCV verbaute Gesichtsdetektor soll auf diesem Bild angewendet werden und die gefundenen Gesichter sollen visualisiert werden. Die Kaskade selbst kann so erzeugt werden

    face_cascade = cv.CascadeClassifier(HAAR_CASCADE)
  
Um die Kaskade dann auf ein Bild anzuwenden muß das Bild zunächst in ein Graustufenbild umgerechnet werden.

      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

Die Kaskade selbst wird dann über diesen Aufruf ausgeführt

      faces = face_cascade.detectMultiScale(gray, 1.3, 5)

Zur Kontrolle während der Aufnahme soll das Live-Bild angezeigt werden und für jedes erkannte Gesicht ein blaues Rechteck um das Gesicht gemalt werden. 

Falls mindestens ein Gesicht im Bild gefunden wurde soll das aufgenommene Rohbild zusammen mit den Koordinaten der Gesichter auf die Festplatte geschrieben werden. Dabei soll das Bild als .PNG Bild gespeichert werden, z.B. mittels

    cv2.imwrite(...)

Eine bis auf die Dateieindung gleichnamige CSV Datei soll parallel auf die Festplatte geschrieben werden um die Koordinaten der in dem Bild gefunden Gesichter zu speichern. Dazu kann der CSVWriter in Python verwendet werden

    with open(os.path.join(self.root, fname + ".csv"), "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            for x, y, w, h in faces:
                writer.writerow([x, y, w, h])

Wenn ein Bild gespeichert wurde sollen für die nächsten 30 Bilder der Webcam keine Speicherung erfolgen, unabhängig davon ob ein Gesicht erkannt wurde oder nicht. So soll sichergestellt werden das die aufgezeichneteten Bilder ein gewisses Mindestmaß an Variabilität beinhalten und nicht zu schnell hintereinander aufgenommen wurden. 

Die Gesichter sollen in einen Ordner "objects" gespeichert werden und dort wiederum in einem Unterordner, dessen Name über die Kommandozeile festgelegt werden kann. Bei obigem Aufruf sollen also alle aufgezeichneten Bilder in dem Ordner objects/dennis landen. Durch diesen Mechanismus soll es sehr einfach von aussen möglich sein die Zielperson zu bennen und für diese dann geordnete Trainigsdaten aufzunehmen und zu speichern. 

Wenn bestimmte Ordner noch nicht auf der Festplatte vorhanden sind soll das Skript sie anlegen. Dies kann mit

    os.path.exists(...)

überprüft und mit 
    
    os.mkdir(...)

erzeugt werden.

### Cropping (Ausschneiden)
Mit dem Aufruf

    python main.py crop --border 0.2 --split 0.3

sollen alle auf der Festplatte gespeicherten Ordner von Personen durchsucht werden und die dort gefunden Gesichter sollen ausgeschnitten werden. Dazu kann mittels

    os.walk(...)
    
die Verzeichnissstruktur der Festplatte durchsucht werden. Die rohen Kamerabilder müssen zunächst geladen werden, z.b. mittels

    frame = cv2.imread(...)

und dann durch einen gespiegelten Rand erweitert werden. Dazu kann die Funktion

    frame = cv.copyMakeBorder(
          frame, top_border, top_border, left_border, left_border, cv.BORDER_REFLECT
      )

verwendet werden. Hintergrund ist das die Gesichter mit einem variablen Rand ausgeschnitten werden sollen. Der Rand (border) ist als prozentualer Faktor der Breite/Höhe über die Kommandozeile steuerbar, im oben genannten Beispiel also 20%. Das bedeutet das ein Gesicht, welches 200 Pixel breit ist, insgesamt 40 Pixel Rand angefügt bekommen soll. Dabei sollen 20 Pixel links und 20 Pixel rechts bzw oben und unten angefügt werden. Da die Gesichter inklusive Rand den tatsächlich vorhandenen Bereich des Kamerabildes verlassen können muß vor dem eigentlichen Ausschneiden entsprechend Rand angefügt werden.

Die ausgeschnittenen Gesichtsbilder sollen aus zwei Ordner "train" und "val" aufgeteilt werden. Innerhalb dieser Ordner sollen entsprechende Unterordner angelegt werden die namentlich identisch mit dem Quellordner sind. Das bedeutet ein Bild welches z.B. aus dem Order "objects/dennis" ausgeschnitten wurde soll im ordner "train/dennis" bzw. "val/dennis" landen. Bis auf das Anfügen des Randes soll das Bild in keinsterweise verändert werden.

Der Anteil der Bilder die im "train" bzw. "val" Ordner landen soll über die Kommandozeile steuerbar sein. Der 

    --split 0.3

Befehl soll angeben das 30% der Bilder in "val" landen während die restlichen 70% der Bilder ein "train" landen. Eine zufällige Auswahl der Bilder ist in Ordnung. 

Wieder sollen fehlende Ordner neu angelegt werden. Ausserdem sollen vor dem eigentlichen Ausschneiden eventuell noch vorhandene Bilder gelöscht werden. Dies kann mittels

    os.remove(os.path.join(root, name))

passieren. 

### Training
Für das Training eines neuronalen Netzwerkes ist ein Großteil des Codes schon vorgegebenen. In der

    train.py

Datei befindet sich der eigentliche Trainingsloop. Das Netzwerk
selbst muß jedoch in der Datei

    network.py

von den Studierenden definiert werden. Hier soll eine geeignete Netzwerkarchitektur gewählt werden die das Gesichtsbilder klassifizieren kann. Die Bilder werden vor dem Training auf eine einheitliche Größe von 256x256 Pixeln skaliert. Als Ausgabe des Netzwerkes muß ein passend dimensionierter Klassenvektor erzeugt werden. Die Anzahl der zu unterscheidenden Klassen wird dabei dem Konstruktor der Netzwerkes übergeben und kann verwendet werden um die Architektur entsprechend einzustellen. 

    class Net(nn.Module):
        def __init__(self, nClasses):
            super().__init__()

In der Datei 

    balancedaccuracy.py 

muß eine Klasse implementiert werden welche die s.g. "Balanced Accuracy" berechnet. Diese ist als die mittlere Erkennungsrate jeder einzelnen Klasse definiert (vergleiche EKI Vorlesung). In der Klasse müssen mehrere Methoden sinnvoll implementiert werden.

In der Reset-Methode müssen interne Zustände so zurückgesetzt werden das ein neuer Durchlauf (in diesem Fall: eine neue Epoche) gestartet werden kann. 
    
    def reset(self):
      # TODO: Reset internal states.
      # Called at the beginning of each epoch


In der Update-Methode müssen die internen Zustände dann so verändert werden das die balanced accuracy berechnet werden kann.

    def update(self, predictions, groundtruth):
        # TODO: Implement the update of internal states
        # based on current network predictios and the groundtruth value.
        #
        # Predictions is a Tensor with logits (non-normalized activations)
        # It is a BATCH_SIZE x N_CLASSES float Tensor. The argmax for each samples
        # indicated the predicted class.
        #
        # Groundtruth is a BATCH_SIZE x 1 long Tensor. It contains the index of the
        # ground truth class.

Die Update-Methode wird innerhalb einer Epoch für jeden Batch aufgerufen. Überlegen Sie welche Informationen sie speichern müssen um beim folgenden Aufruf der getBACC Methode die "balanced accuracy" berechnen und zurückgeben zu können.

    def getBACC(self):
        # TODO: Calculcate and return balanced accuracy 
        # based on current internal state

### Live
Mit dem Aufruf

    python main.py live --border 0.2

soll das Modell live ausgeführt werden. Ähnlich wie im Record-Modus soll ein Live-Bild der Webcam aufgenommen werden. Auch die Haar-Wavelet Kaskade soll wieder ausgeführt werden. Laden Sie den gespeicherten CheckPoint des Modells
und instantieren Sie das PyTorch Modell.

Das Bild muß ebenfalls wieder mit einem Rand versehen werden da auch diesmal wieder alle Detektionen ausgeschnitten werden müssen. Achten Sie darauf die Bilder korrekt mit Rand auszuschneiden da das Netzwerk ja ebenfalls mit diesem Rand gelernt hat.

Wandeln Sie jeden Gesichtsausschnitt in ein "Image" aus der PIL-Bibliothek um und verwenden Sie dann die ValidationTransform aus der transforms.py Datei um das Bild in einen PyTorch Tensor umzurechnen. Verwenden Sie das geladenen Netzwerk um den Bildauschnitt zu klassifizieren und zeichnen Sie wieder ein Rechteck um das Gesicht. Schreiben Sie den Namen der erkannten Person über das Rechteck.

    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.rectangle(frame, (x, y - 32), (x + w, y), (255, 0, 0), -1)
    cv.putText(
      frame, name, (x + 8, y - 6), font, 1, (255, 255, 255), 2, cv.    LINE_AA
    )


