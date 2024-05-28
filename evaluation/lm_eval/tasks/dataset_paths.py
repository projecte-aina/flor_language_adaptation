import os


main_directory = str(os.path.abspath(os.path.join(os.getcwd())))

dataset_paths = {
        "belebele": main_directory+"/data_offline/belebele/belebele.py",
        "xquad": main_directory+"/data_offline/xquad/xquad.py",
        "catalanqa": main_directory+"/data_offline/catalanqa/catalanqa.py",
        "coqcat": main_directory+"/data_offline/coqcat/coqcat.py",
        "xnli": main_directory+"/data_offline/xnli/xnli.py",
        "xnli_ca": main_directory+"/data_offline/xnli-ca/xnli_ca.py",
        "teca": main_directory+"/data_offline/teca/teca.py",
        "paws-x": main_directory+"/data_offline/paws-x/paws-x.py",
        "parafraseja": main_directory+"/data_offline/parafraseja/Parafraseja.py",
        "xstorycloze": main_directory+"/data_offline/xstorycloze/xstory_cloze.py",
        "copa": main_directory+"/data_offline/COPA/copa.py",
        "copa_ca": main_directory+"/data_offline/copa_ca/COPA-ca.py",
        "flores": main_directory+"/data_offline/flores/flores.py",
        "wnli_es": main_directory+"/data_offline/wnli-es/wnli-es.py",
        "wnli_ca": main_directory+"/data_offline/wnli-ca/wnli-ca.py",
        "cabreu": main_directory+"/data_offline/cabreu/caBreu.py",
        "ai2_arc": main_directory+"/data_offline/ai2_arc/ai2_arc.py",
        "wikilingua": main_directory+"/data_offline/wikilingua/wikilingua.py",
        "hellaswag": main_directory+"/data_offline/hellaswag/hellaswag.py",
        "winogrande": main_directory+"/data_offline/winogrande/winogrande.py",
        "glue": main_directory+"/data_offline/glue/glue.py",
        "lambada_openai": main_directory+"/data_offline/lambada_openai/lambada_openai.py",
        "piqa": main_directory+"/data_offline/piqa/piqa.py",
        "squad": main_directory+"/data_offline/squad/squad.py",
        "openbookqa": main_directory+"/data_offline/openbookqa/openbookqa.py",
        "sacrebleu": main_directory+"/data_offline/sacrebleu/"
                }

