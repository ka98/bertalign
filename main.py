from bertalign import Bertalign
import pickle
import os.path
import subprocess

if not os.path.isfile("aligned.txt"): 
    with open("literature/jos/to_be_aligned/nb_jos_01.md") as file:
        src = file.read()

    with open("literature/jos/to_be_aligned/de_jos_01.md") as file:
        tgt = file.read()

    aligner = Bertalign("no", "de", src, tgt, src_language_code="nb-NO", tgt_language_code="de-DE", output_file="jos", title="Gesammelte Schriften - Band 1 - Johan O. Smith")
    aligner.align_sents()
    file = open("aligned.txt", 'wb')
    pickle.dump(aligner, file)
else:
    file = open("aligned.txt", 'rb')
    aligner = pickle.load(file)
aligner.output_tmx()
aligner.output_excel()
# aligner.print_sents()

subprocess.call(["sed -i 's/<!\[CDATA\[//g' German_jos.tmx"], shell=True)
subprocess.call(["sed -i 's/\]\]>//g' German_jos.tmx"], shell=True)
subprocess.call(["sed -i 's@<header creationtool=\"Translate Toolkit\" creationtoolversion=\"3.12.2\" segtype=\"sentence\" o-tmf=\"UTF-8\" adminlang=\"en\" srclang=\"nb-NO\" datatype=\"PlainText\"/>@<header creationtool=\"memsource_tool\" creationtoolversion=\"2.0\" segtype=\"sentence\" o-tmf=\"memsourceTM\" adminlang=\"en\" srclang=\"nb-no\" datatype=\"unknown\" creationdate=\"20240828T154242Z\"/>@g' German_jos.tmx"], shell=True)