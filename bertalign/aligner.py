import numpy as np
import os
import re

from bertalign import model
from bertalign.corelib import *
from bertalign.utils import *
from translate.storage.tmx import tmxfile

from pandas import DataFrame

from lxml import etree
from translate.storage.tmx import tmxfile, tmxunit
from translate.misc.xml_helpers import setXMLlang

# Patching TMX Creation Plugin
def addtranslation_patch(self, source, srclang, translation, translang, comment=None, context_prev=None, context_next=None, filename=None):
        """Addtranslation method for testing old unit tests."""
        unit = self.addsourceunit(source)
        unit.target = translation
        if comment is not None and len(comment) > 0:
            unit.addnote(comment)

        tuvs = unit.xmlelement.iterdescendants(self.namespaced("tuv"))
        src_tuv = next(tuvs)
        setXMLlang(src_tuv, srclang)
        
        if context_prev is not None and len(context_prev) > 0:
            context_prev_element = etree.SubElement(src_tuv, self.namespaced("prop"), {"type": "context_prev"})
            context_prev_element.text = context_prev.strip()
            
        if context_next is not None and len(context_next) > 0:
            context_next_element = etree.SubElement(src_tuv, self.namespaced("prop"), {"type": "context_next"})
            context_next_element.text = context_next.strip()
        
        tgt_tuv = next(tuvs)
        setXMLlang(tgt_tuv, translang)
        
        if filename is not None and len(filename) > 0:
            filename_element = etree.SubElement(tgt_tuv, self.namespaced("prop"), {"type": "filename"})
            filename_element.text = filename.strip()
        
# def addcontext_prev_patch(self, text, origin=None, position="append"):
#         """
#         Add a context_prev property.

#         The origin parameter is ignored
#         """
#         context_prev = etree.SubElement(self.xmlelement, self.namespaced("prop"), {"type": "context_prev"})
#         context_prev.text = text.strip()
        
# def addcontext_next_patch(self, text, origin=None, position="append"):
#         """
#         Add a context_next property.

#         The origin parameter is ignored
#         """
#         context_next = etree.SubElement(self.xmlelement, self.namespaced("prop"), {"type": "context_next"})
#         context_next.text = text.strip()
        
# def addfilename_patch(self, text, origin=None, position="append"):
#         """
#         Add a context_next property.

#         The origin parameter is ignored
#         """
#         filename = etree.SubElement(self.xmlelement, self.namespaced("prop"), {"type": "filename"})
#         filename.text = text.strip()
        
setattr(tmxfile, "addtranslation", addtranslation_patch)
# setattr(tmxunit, "addcontextprev", addcontext_prev_patch)
# setattr(tmxunit, "addcontextnext", addcontext_next_patch)
# setattr(tmxunit, "addfilename", addfilename_patch)

class Bertalign:
    def __init__(self,
                 src_lang,
                 tgt_lang,
                 src,
                 tgt,
                 max_align=5,
                 top_k=3,
                 win=5,
                 skip=-0.1,
                 margin=True,
                 len_penalty=True,
                 is_split=False,
                 src_language_code="nb-NO",
                 tgt_language_code="en-US",
                 output_file="_output",
                 title=""
               ):
        
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        
        src = clean_text(src)
        tgt = clean_text(tgt)
        # src_lang = detect_lang(src)
        # tgt_lang = detect_lang(tgt)
        
        if is_split:
            src_sents = src.splitlines()
            tgt_sents = tgt.splitlines()
        else:
            src_sents = split_sents(src, src_lang)
            tgt_sents = split_sents(tgt, tgt_lang)
 
        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        
        src_lang = LANG.ISO[src_lang]
        tgt_lang = LANG.ISO[tgt_lang]
        
        print("Source language: {}, Number of sentences: {}".format(src_lang, src_num))
        print("Target language: {}, Number of sentences: {}".format(tgt_lang, tgt_num))

        print("Embedding source and target text using {} ...".format(model.model_name))
        src_vecs, src_lens = model.transform(src_sents, max_align - 1)
        tgt_vecs, tgt_lens = model.transform(tgt_sents, max_align - 1)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs
        
        self.src_language_code = src_language_code
        self.tgt_language_code = tgt_language_code
        self.output_file = output_file
        self.title = title
        
    def align_sents(self):

        print("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.src_vecs[0,:], self.tgt_vecs[0,:], k=self.top_k)
        first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I)
        first_alignment = first_back_track(self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types)
        
        print("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        second_pointers = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                            second_w, second_path, second_alignment_types,
                                            self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty)
        second_alignment = second_back_track(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types)
        
        print("Finished! Successfully aligning {} {} sentences to {} {} sentences\n".format(self.src_num, self.src_lang, self.tgt_num, self.tgt_lang))
        self.result = second_alignment
    
    def output_tmx(self):
        tm_file = tmxfile(None, self.src_language_code, self.tgt_language_code)
        H1 = ""
        H2 = ""
        H3 = ""
        H4 = ""
        H5 = ""
        H6 = ""
        for idx, bead in enumerate(self.result):
            
            prev_segment = None
            next_segment = None
            
            if idx != 0:
                prev_segment = re.sub(r'^#+\s+', '', self._get_line(self.result[idx-1][0], self.src_sents)).replace("*", "")
                pass
            if idx < len(self.result) - 1:
                next_segment = re.sub(r'^#+\s+', '', self._get_line(self.result[idx+1][0], self.src_sents)).replace("*", "")
                pass
            
            src_text = self._get_line(bead[0], self.src_sents)
            tgt_text = self._get_line(bead[1], self.tgt_sents)
            
            if tgt_text.startswith("#"):
                if tgt_text.startswith("##"):
                    if tgt_text.startswith("###"):
                        if tgt_text.startswith("####"):
                            if tgt_text.startswith("#####"):
                                if tgt_text.startswith("######"):
                                    H6 = re.sub(r'^#+\s+', '', tgt_text)
                                else:
                                    H5 = re.sub(r'^#+\s+', '', tgt_text)
                                    H6 = ""
                            else:
                                H4 = re.sub(r'^#+\s+', '', tgt_text)
                                H5 = ""
                                H6 = ""
                        else:
                            H3 = re.sub(r'^#+\s+', '', tgt_text)
                            H4 = ""
                            H5 = ""
                            H6 = ""
                    else:
                        H2 = re.sub(r'^#+\s+', '', tgt_text)
                        H3 = ""
                        H4 = ""
                        H5 = ""
                        H6 = ""
                else:
                    H1 = re.sub(r'^#+\s+', '', tgt_text)
                    H2 = ""
                    H3 = ""
                    H4 = ""
                    H5 = ""
                    H6 = ""
                    
            src_text = re.sub(r'^#+\s+', '', src_text).replace("*", "")
            tgt_text = re.sub(r'^#+\s+', '', tgt_text).replace("*", "")
            
            # i = 1
            # x = 1
            
            # # check if Markdown annotation, only Emph is supported for now!
            # if src_text.count("*") > 0:
            #     if src_text.count("*") % 2:
            #         # even, so we have to inject notation
            #         while src_text.count("*") > 1:
            #             src_text = src_text.replace('*', f'<bpt i="{i}" x = "{x}" type="italic"\><i>\</bpt\>', 1)
            #             src_text = src_text.replace('*', f'<ept i="{i}" x = "{x}" type="italic"\></i>\</ept\>', 1)
            #             i = i + 1
            #             x = x + 1
            #         #<bpt i="1000001" x="1000001" type="formatting">{b&gt;</bpt>
            #         #<ept i="1000001">&lt;b}</ept>
            #     else:
            #         src_text = src_text.replace("*", "")
                 
            # i = 1
            # x = 1
                    
            # if tgt_text.count("*") > 0:
            #     if tgt_text.count("*") % 2:
            #         # even, so we have to inject notation
            #         while tgt_text.count("*") > 1:
            #             tgt_text = tgt_text.replace('*', f'<bpt i="{i}" x = "{x}" type="italic"\><i>\</bpt\>', 1)
            #             tgt_text = tgt_text.replace('*', f'<ept i="{i}" x = "{x}" type="italic"\></i>\</ept\>', 1)
            #         #<bpt i="1000001" x="1000001" type="formatting">{b&gt;</bpt>
            #         #<ept i="1000001">&lt;b}</ept>
            #     else:
            #         tgt_text = tgt_text.replace("*", "")
            src_text = src_text.replace('<span class="calibre16">', f'<btp i="1" x="1" type="ulined">')
            # Replace the closing span tag with </ept> including attributes and formatting type
            src_text = src_text.replace('</span>', f'</ept i="1" x="1" type="ulined">')
            
            src_text = etree.CDATA(src_text)
            
            filename = self.title
            if H1:
                filename += f" - {H1}"
            if H2:
                filename += f" - {H2}"
            if H3:
                filename += f" - {H3}"
            if H4:
                filename += f" - {H4}"
            if H5:
                filename += f" - {H5}"
            if H6:
                filename += f" - {H6}"
            
            tm_file.addtranslation(
                src_text,
                self.src_language_code,
                tgt_text,
                self.tgt_language_code,
                context_prev=prev_segment,
                context_next=next_segment,
                filename=filename
                
            )
        with open(f"{self.tgt_lang}_{self.output_file}.tmx", "wb") as output:
            tm_file.serialize(output)
            
    def output_excel(self):
        list = []
        for bead in (self.result):
            list.append(
                (self._get_line(bead[0], self.src_sents),
                self._get_line(bead[1], self.tgt_sents))
            )
        df = DataFrame(list, columns=[self.src_language_code, self.tgt_language_code])
        df.to_excel(f"{self.tgt_lang}_{self.output_file}.xlsx", sheet_name='sheet1', index=False)
    
    def print_sents(self):
        for bead in (self.result):
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(src_line + "\n" + tgt_line + "\n")

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1]+1])
        return line
