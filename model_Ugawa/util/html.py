import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir_other = os.path.join(self.web_dir, 'other_images')
        self.img_dir_fake_B = os.path.join(self.web_dir, 'fake_B')
        self.img_dir_real_A = os.path.join(self.web_dir, 'real_A')
        self.img_dir_real_B = os.path.join(self.web_dir, 'real_B')

        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir_other):
            os.makedirs(self.img_dir_other)
        if not os.path.exists(self.img_dir_fake_B):
            os.makedirs(self.img_dir_fake_B)
        if not os.path.exists(self.img_dir_real_A):
            os.makedirs(self.img_dir_real_A)
        if not os.path.exists(self.img_dir_real_B):
            os.makedirs(self.img_dir_real_B)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir_other, self.img_dir_fake_B, self.img_dir_real_A, self.img_dir_real_B

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            if 'fake_B' in link:
                                img_type = 'fake_B'
                            elif 'real_A' in link:
                                img_type = 'real_A'
                            elif 'real_B' in link:
                                img_type = 'real_B'
                            else:
                                img_type = 'other_images'
                            with a(href=os.path.join(img_type, link)):
                                img(style="width:%dpx" % width, src=os.path.join(img_type, im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
