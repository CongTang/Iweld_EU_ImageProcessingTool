from merge_window import Ui_merge
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtCore import QUrl
import sys
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from PyQt5.QtGui import QImage,QPixmap,QDoubleValidator
from PyQt5 import QtWidgets, QtCore, QtGui
import cgitb
cgitb.enable()
import cv2
import numpy as np
version = cv2.__version__

class Browser(QMainWindow):
    def __init__(self):
        super(Browser, self).__init__()
        self.setWindowTitle("browser")
        self.setGeometry(80,80,900,900)
        self._browser=QWebEngineView()
        self._browser.load(QUrl("https://www.mrpd-jamesren-wixsite.com/iweld-eu"))
        self.setCentralWidget(self._browser)

class mwindow(QMainWindow, Ui_merge):
    def __init__(self):
        super(mwindow, self).__init__()
        self.setupUi(self)  # 初始化窗口显示控件
        self._init()

    def _init(self):
        #公共
        self.img = None
        self.size_scale = 1


        self.img_hsv = None
        self.hsv_value_list = []
        self.flood_point_list = []
        self.area_line_point_info = []
        self.fig_area_list = []

        self.pushButton_open_image.clicked.connect(self.open_image)
        self.pushButton_hsv_cancle.clicked.connect(self.hsv_cancle)
        self.pushButton_hsv_set_value.clicked.connect(self.hsv_set_value)
        self.pushButton_flood_cancle.clicked.connect(self.flood_cancle)
        self.pushButton_open_data_file.clicked.connect(self.open_data_file)
        self.pushButton_more_info.clicked.connect(self.open_browser)
        self.pushButton_exit.clicked.connect(sys.exit)

        self.label_hsv_src_image.mousePressEvent = self.getMousePos_hsv
        self.label_flood_src_image.mousePressEvent = self.getMousePos_flood

        doubleValidator = QDoubleValidator(self)
        doubleValidator.setRange(1, 10000000000, 3)
        self.lineEdit_img_w.setValidator(doubleValidator)
        self.lineEdit_hsv_la.setValidator(doubleValidator)
        self.lineEdit_hsv_ha.setValidator(doubleValidator)
        self.lineEdit_bi_la.setValidator(doubleValidator)
        self.lineEdit_bi_ha.setValidator(doubleValidator)

        self.lineEdit_fig_area_increment.setValidator(doubleValidator)
        self.lineEdit_fig_line_increment.setValidator(doubleValidator)
        self.lineEdit_fig_area_min_value.setValidator(doubleValidator)
        self.lineEdit_fig_area_max_value.setValidator(doubleValidator)
        self.lineEdit_fig_line_min_value.setValidator(doubleValidator)
        self.lineEdit_fig_line_max_value.setValidator(doubleValidator)

        #hsv窗口绑定事件
        self.horizontalSlider_lh.valueChanged.connect(self.valChange_lh)
        self.horizontalSlider_ls.valueChanged.connect(self.valChange_ls)
        self.horizontalSlider_lv.valueChanged.connect(self.valChange_lv)
        self.horizontalSlider_hh.valueChanged.connect(self.valChange_hh)
        self.horizontalSlider_hs.valueChanged.connect(self.valChange_hs)
        self.horizontalSlider_hv.valueChanged.connect(self.valChange_hv)
        self.checkBox_hsv_inv.stateChanged.connect(self.update_hsv_run)
        self.checkBox_hsv_smooth.stateChanged.connect(self.update_hsv_run)
        self.lineEdit_hsv_la.textChanged.connect(self.update_hsv_run)
        self.lineEdit_hsv_ha.textChanged.connect(self.update_hsv_run)
        self.pushButton_hsv_export.clicked.connect(self.export_data)

        #二值化窗口绑定事件
        self.horizontalSlider_bi.valueChanged.connect(self.valChange_bi)
        self.checkBox_bi_inv.stateChanged.connect(self.update_bi_run)
        self.checkBox_bi_smooth.stateChanged.connect(self.update_bi_run)
        self.lineEdit_bi_la.textChanged.connect(self.update_bi_run)
        self.lineEdit_bi_ha.textChanged.connect(self.update_bi_run)
        self.pushButton_bi_export.clicked.connect(self.export_data)

        #边缘检测窗口绑定事件
        self.horizontalSlider_t1.valueChanged.connect(self.valChange_t1)
        self.horizontalSlider_t2.valueChanged.connect(self.valChange_t2)
        self.lineEdit_edge_la.textChanged.connect(self.update_edge_run)
        self.lineEdit_edge_ha.textChanged.connect(self.update_edge_run)
        self.pushButton_edge_export.clicked.connect(self.export_data)

        #漫水算法窗口绑定事件
        self.horizontalSlider_diff.valueChanged.connect(self.valChange_diff)
        self.pushButton_flood_export.clicked.connect(self.export_data)

        #画图窗口
        # self.lineEdit_fig_area_increment.textChanged.connect(self.draw)
        # self.lineEdit_fig_line_increment.textChanged.connect(self.draw)
        # self.lineEdit_fig_area_min_value.textChanged.connect(self.draw)
        # self.lineEdit_fig_area_max_value.textChanged.connect(self.draw)
        # self.lineEdit_fig_line_min_value.textChanged.connect(self.draw)
        # self.lineEdit_fig_line_max_value.textChanged.connect(self.draw)
        self.pushButton_fig_draw.clicked.connect(self.draw)

        self.lineEdit_img_w.textChanged.connect(self.lineEdit_img_change)
        self.tabWidget.currentChanged.connect(self.change_page)

    def lineEdit_img_change(self):
        self.change_page(self.tabWidget.currentIndex())

    def change_page(self,index):
        if self.img is not  None:
            if index==0:
                showImage = QImage(self.img.data, self.img.shape[1], self.img.shape[0],
                                       self.img.shape[1] * 3,
                                       QImage.Format_RGB888).rgbSwapped()
                self.label_hsv_src_image.setPixmap(QPixmap.fromImage(showImage))
                self.update_hsv_run()
            elif index==1:
                showImage = QImage(self.img.data, self.img.shape[1], self.img.shape[0],
                                   self.img.shape[1] * 3,
                                   QImage.Format_RGB888).rgbSwapped()
                self.label_bi_src_image.setPixmap(QPixmap.fromImage(showImage))
                self.update_bi_run()

            elif index==2:
                showImage = QImage(self.img.data, self.img.shape[1], self.img.shape[0],
                                   self.img.shape[1] * 3,
                                   QImage.Format_RGB888).rgbSwapped()
                self.label_edge_src_image.setPixmap(QPixmap.fromImage(showImage))
                self.update_edge_run()

            elif index==3:
                showImage = QImage(self.img.data, self.img.shape[1], self.img.shape[0],
                                   self.img.shape[1] * 3,
                                   QImage.Format_RGB888).rgbSwapped()
                self.label_flood_src_image.setPixmap(QPixmap.fromImage(showImage))
                self.update_flood_run()

    def open_image(self):
        #读取本地图片开始执行
        image_path, _ = QFileDialog.getOpenFileName(self, 'open file', ".",
                                                    ("Images (*.png *.bmp *.jpg)"))
        if image_path != "":
            self.img = cv2.imread(image_path)
            self.src_h, self.src_w = self.img.shape[:2]
            self.img_area = self.src_w * self.src_h
            self.lineEdit_hsv_ha.setText(str(self.img_area))
            self.img_hsv = cv2.cvtColor(self.img,cv2.COLOR_BGR2HSV)
            #清空
            self.hsv_value_list = []
            self.flood_point_list = []
            self.area_line_point_info = []
            self.change_page(self.tabWidget.currentIndex())


    def set_image_src(self, frame):

        hsv_showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1] * 3,
                         QImage.Format_RGB888).rgbSwapped()

        bi_showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1] * 3,
                         QImage.Format_RGB888).rgbSwapped()
        self.label_hsv_src_image.setPixmap(QPixmap.fromImage(hsv_showImage))
        self.label_bi_src_image.setPixmap(QPixmap.fromImage(bi_showImage))



    """滚动条数值滑动触发函数区域"""

    def valChange_lh(self):
        self.label_lh.setText("L_H:{}".format(self.horizontalSlider_lh.value()))
        self.update_hsv_run()

    def valChange_ls(self):
        self.label_ls.setText("L_S:{}".format(self.horizontalSlider_ls.value()))
        self.update_hsv_run()

    def valChange_lv(self):
        self.label_lv.setText("L_V:{}".format(self.horizontalSlider_lv.value()))
        self.update_hsv_run()

    def valChange_hh(self):
        self.label_hh.setText("H_H:{}".format(self.horizontalSlider_hh.value()))
        self.update_hsv_run()

    def valChange_hs(self):
        self.label_hs.setText("H_S:{}".format(self.horizontalSlider_hs.value()))
        self.update_hsv_run()

    def valChange_hv(self):
        self.label_hv.setText("H_V:{}".format(self.horizontalSlider_hv.value()))
        self.update_hsv_run()

    def valChange_bi(self):
        self.label_bi.setText("B:{}".format(self.horizontalSlider_bi.value()))
        self.update_bi_run()

    def valChange_t1(self):
        self.label_t1.setText("t1:{}".format(self.horizontalSlider_t1.value()))
        self.update_edge_run()

    def valChange_t2(self):
        self.label_t2.setText("t2:{}".format(self.horizontalSlider_t2.value()))
        self.update_edge_run()

    def valChange_diff(self):
        self.label_diff.setText("diff:{}".format(self.horizontalSlider_diff.value()))
        self.update_flood_run()

    #鼠标位置绑定事件
    def getMousePos_hsv(self, event):
        if self.img_hsv is not None:
            x = event.pos().x()
            y = event.pos().y()
            h, w = self.img.shape[:2]
            real_x, real_y = int(x * w / self.label_hsv_src_image.width()), int(
                y * h / self.label_hsv_src_image.height())
            self.hsv_value_list.append(self.img_hsv[real_y, real_x].tolist())
            self.update_click_hsv_value()

    def getMousePos_flood(self, event):
        if self.img is not None:
            x = event.pos().x()
            y = event.pos().y()
            h, w = self.img.shape[:2]
            real_x, real_y = int(x * w / self.label_flood_src_image.width()), int(
                y * h / self.label_flood_src_image.height())
            self.flood_point_list.append([real_x, real_y])
            self.update_flood_run()

    #hsv
    def update_click_hsv_value(self):
        self.textEdit_hsv_info.clear()
        if len(self.hsv_value_list)==0:
            self.label_hsv_value.setText("hsv::")
            self.label_lh_value.setText("0")
            self.label_ls_value.setText("0")
            self.label_lv_value.setText("0")
            self.label_hh_value.setText("0")
            self.label_hs_value.setText("0")
            self.label_hv_value.setText("0")
        else:
            click_hsv_value = self.hsv_value_list[-1]
            self.label_hsv_value.setText("hsv:{}".format(click_hsv_value))
            hsv_value_list = np.array(self.hsv_value_list)
            self.hsv_lh = np.min(hsv_value_list[:,0])
            self.hsv_ls = np.min(hsv_value_list[:,1])
            self.hsv_lv = np.min(hsv_value_list[:,2])
            self.hsv_hh = np.max(hsv_value_list[:,0])
            self.hsv_hs = np.max(hsv_value_list[:,1])
            self.hsv_hv = np.max(hsv_value_list[:,2])
            self.label_lh_value.setText(str(self.hsv_lh))
            self.label_ls_value.setText(str(self.hsv_ls))
            self.label_lv_value.setText(str(self.hsv_lv))
            self.label_hh_value.setText(str(self.hsv_hh))
            self.label_hs_value.setText(str(self.hsv_hs))
            self.label_hv_value.setText(str(self.hsv_hv))

            for i in self.hsv_value_list:
                self.textEdit_hsv_info.append(str(i))

    def hsv_cancle(self):
        if len(self.hsv_value_list)>1:
            self.hsv_value_list = self.hsv_value_list[:-1]
        else:
            self.hsv_value_list=[]
        self.update_click_hsv_value()



    def hsv_set_value(self):
        if len(self.hsv_value_list)>0:
            self.horizontalSlider_lh.setValue(self.hsv_lh)
            self.horizontalSlider_ls.setValue(self.hsv_ls)
            self.horizontalSlider_lv.setValue(self.hsv_lv)
            self.horizontalSlider_hh.setValue(self.hsv_hh)
            self.horizontalSlider_hs.setValue(self.hsv_hs)
            self.horizontalSlider_hv.setValue(self.hsv_hv)
            self.label_lh.setText("L_H:{}".format(self.horizontalSlider_lh.value()))
            self.label_ls.setText("L_S:{}".format(self.horizontalSlider_ls.value()))
            self.label_lv.setText("L_V:{}".format(self.horizontalSlider_lv.value()))
            self.label_hh.setText("H_H:{}".format(self.horizontalSlider_hh.value()))
            self.label_hs.setText("H_S:{}".format(self.horizontalSlider_hs.value()))
            self.label_hv.setText("H_V:{}".format(self.horizontalSlider_hv.value()))


    def update_hsv_run(self):

        if self.img_hsv is None:
            return

        if len(self.lineEdit_img_w.text()):
            self.size_scale = float(self.lineEdit_img_w.text()) / self.src_w
        else:
            self.size_scale = 1

        self.area_line_point_info = []
        #使用hsv 阈值过滤
        mask = cv2.inRange(self.img_hsv, (self.horizontalSlider_lh.value(), self.horizontalSlider_ls.value(), self.horizontalSlider_lv.value()),
                           (self.horizontalSlider_hh.value(), self.horizontalSlider_hs.value(), self.horizontalSlider_hv.value()))


        if  self.checkBox_hsv_smooth.isChecked():
            # 腐蚀的作用是消除物体边界点，使目标缩小，可以消除小于结构元素的噪声点,闭运算时先膨胀后腐蚀的过程，可以填充物体内细小的空洞，并平滑物体边界。
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                             iterations=1)

            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                 iterations=1)
        if self.checkBox_hsv_inv.isChecked():
            mask = cv2.bitwise_not(mask)

        # 找出掩码轮廓点
        if version.startswith("3"):  # opencv 3版本，CHAIN_APPROX_SIMPLE保留所有轮廓上的点
            _, contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        elif version.startswith("4"):  # opencv 4版本
            contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raise Exception("版本不支持:{}".format(version))

        draw_image = self.img.copy()
        # 遍历轮廓

        area_id = 0
        area_info = ""

        for c in contours:
            # 计算轮廓包围的面积
            area = cv2.contourArea(c)
            if area / self.img_area > 0.99:  # 面积基本和原图一样大了，一般是把原图最外层认为是一个轮廓了，可以不取，
                continue
            # 换算实际大小比例
            area = area * self.size_scale * self.size_scale
            # 面积过滤
            if area > float(self.lineEdit_hsv_la.text()) and area < float(self.lineEdit_hsv_ha.text()):
                area_id += 1
                # 画出这条轮廓
                cv2.drawContours(draw_image, [c], -1, (0, 0, 255), 1)
                (x_center, y_center), radius = cv2.minEnclosingCircle(c)
                diameter = 2 * radius * self.size_scale
                # 轮廓点维度reshpe 成 n 行 2列，这样每行就是 一个x，y值
                point = np.reshape(c, [-1, 2])
                #在轮廓中心附近写上id
                cv2.putText(draw_image,str(area_id),(int(x_center), int(y_center)),1,self.src_w/150,(255,0,0))
                self.area_line_point_info.append([area,diameter,point.tolist()])
                area_info += "id:{},area:{}\n".format(area_id, area)
            else:
                cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)
        self.textEdit_hsv_info.setText(area_info)
        self.set_hsv_mask(mask)
        self.set_hsv_image_draw(draw_image)

    def set_hsv_mask(self,frame):
        self.tabWidget.currentIndex()
        showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1],
                         QImage.Format_Grayscale8)
        # showImage = showImage.scaled(self.image_hsv_mask.width(),
        #                            self.image_hsv_mask.height())
        self.image_hsv_mask.setPixmap(QPixmap.fromImage(showImage))

    def set_hsv_image_draw(self, frame):

        showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1] * 3,
                         QImage.Format_RGB888).rgbSwapped()
        # showImage = showImage.scaled(self.image_hsv_draw.width(),
        #                            self.image_hsv_draw.height())
        self.image_hsv_draw.setPixmap(QPixmap.fromImage(showImage))

    #gray
    def update_bi_run(self):
        if self.img is None:
            return

        if len(self.lineEdit_img_w.text()):
            self.size_scale = float(self.lineEdit_img_w.text()) / self.src_w
        else:
            self.size_scale = 1

        self.area_line_point_info =[]
        imggray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        mask = cv2.threshold(imggray, self.horizontalSlider_bi.value(), 255, cv2.THRESH_BINARY)[1]


        if  self.checkBox_bi_smooth.isChecked():
            # 腐蚀的作用是消除物体边界点，使目标缩小，可以消除小于结构元素的噪声点,闭运算时先膨胀后腐蚀的过程，可以填充物体内细小的空洞，并平滑物体边界。
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                             iterations=1)

            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                 iterations=1)
        if self.checkBox_bi_inv.isChecked():
            mask = cv2.bitwise_not(mask)

        # 找出掩码轮廓点
        if version.startswith("3"):  # opencv 3版本，CHAIN_APPROX_SIMPLE保留所有轮廓上的点
            _, contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        elif version.startswith("4"):  # opencv 4版本
            contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raise Exception("版本不支持:{}".format(version))

        draw_image = self.img.copy()
        # 遍历轮廓

        area_id = 0
        area_info = ""

        for c in contours:
            # 计算轮廓包围的面积
            area = cv2.contourArea(c)
            if area / self.img_area > 0.99:  # 面积基本和原图一样大了，一般是把原图最外层认为是一个轮廓了，可以不取，
                continue
            # 换算实际大小比例
            area = area * self.size_scale * self.size_scale
            # 面积过滤
            if area > float(self.lineEdit_bi_la.text()) and area < float(self.lineEdit_bi_ha.text()):
                area_id += 1
                # 画出这条轮廓
                cv2.drawContours(draw_image, [c], -1, (0, 0, 255), 1)
                (x_center, y_center), radius = cv2.minEnclosingCircle(c)
                diameter = 2 * radius * self.size_scale
                # 轮廓点维度reshpe 成 n 行 2列，这样每行就是 一个x，y值
                point = np.reshape(c, [-1, 2])
                # 在轮廓中心附近写上id
                cv2.putText(draw_image, str(area_id), (int(x_center), int(y_center)), 1, self.src_w/150, (255, 0, 0))
                self.area_line_point_info.append([area, diameter, point.tolist()])
                area_info += "id:{},area:{}\n".format(area_id, area)
            else:
                cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)
        self.textEdit_bi_info.setText(area_info)
        self.set_bi_mask(mask)
        self.set_bi_image_draw(draw_image)

    def set_bi_mask(self,frame):
        showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1],
                         QImage.Format_Grayscale8)
        # showImage = showImage.scaled(self.image_bi_mask.width(),
        #                            self.image_bi_mask.height())
        self.image_bi_mask.setPixmap(QPixmap.fromImage(showImage))

    def set_bi_image_draw(self, frame):
        showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1] * 3,
                         QImage.Format_RGB888).rgbSwapped()
        # showImage = showImage.scaled(self.image_bi_draw.width(),
        #                            self.image_bi_draw.height())
        self.image_bi_draw.setPixmap(QPixmap.fromImage(showImage))

    #edge
    def update_edge_run(self):
        if self.img is None:
            return

        if len(self.lineEdit_img_w.text()):
            self.size_scale = float(self.lineEdit_img_w.text()) / self.src_w
        else:
            self.size_scale = 1

        self.area_line_point_info = []
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        mask = cv2.Canny(img_gray,self.horizontalSlider_t1.value(),self.horizontalSlider_t2.value(),apertureSize=3)
        # mask = cv2.bitwise_not(mask)
        #给边缘检测后的图片最外层加上一层‘白边，用于解决边缘检测处理图像边缘没有界定线
        mask[:, 0] = 255
        mask[:, -1] = 255
        mask[0, :] = 255
        mask[-1, :] = 255

        # 膨胀腐蚀，边缘检测后的结果图在边界处可能会断线，导致计算误差，需要做膨胀腐蚀，将部分‘断线’ 连接上
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=1)

        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                         iterations=1)

        # 找出掩码轮廓点
        if version.startswith("3"):  # opencv 3版本，CHAIN_APPROX_SIMPLE保留所有轮廓上的点
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        elif version.startswith("4"):  # opencv 4版本
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raise Exception("版本不支持:{}".format(version))

        contours = [i for i in contours if len(i)>2]

        draw_image = self.img.copy()
        area_id = 0
        area_info = ""

        for c in contours:
            # 计算轮廓包围的面积
            area = cv2.contourArea(c)
            if area/self.img_area>0.99:  #面积基本和原图一样大了，一般是把原图最外层认为是一个轮廓了，可以不取，
                continue
            #换算实际大小比例
            area = area*self.size_scale*self.size_scale
            # 面积过滤
            if area > float(self.lineEdit_edge_la.text()) and area < float(self.lineEdit_edge_ha.text()):
                area_id += 1
                # 画出这条轮廓
                cv2.drawContours(draw_image, [c], -1, (0, 0, 255), 1)
                (x_center, y_center), radius = cv2.minEnclosingCircle(c)
                diameter = 2 * radius * self.size_scale
                # 轮廓点维度reshpe 成 n 行 2列，这样每行就是 一个x，y值
                point = np.reshape(c, [-1, 2])
                # 在轮廓中心附近写上id
                cv2.putText(draw_image, str(area_id), (int(x_center), int(y_center)), 1, self.src_w/150, (255, 0, 0))
                self.area_line_point_info.append([area, diameter, point.tolist()])
                area_info += "id:{},area:{}\n".format(area_id, area)
            else:
                cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)
        self.textEdit_edge_info.setText(area_info)
        self.set_edge_mask(mask)
        self.set_edge_image_draw(draw_image)

    def set_edge_mask(self, frame):
        showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1],
                         QImage.Format_Grayscale8)
        # showImage = showImage.scaled(self.image_edge_mask.width(),
        #                            self.image_edge_mask.height())
        self.image_edge_mask.setPixmap(QPixmap.fromImage(showImage))

    def set_edge_image_draw(self, frame):
        showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1] * 3,
                         QImage.Format_RGB888).rgbSwapped()
        # showImage = showImage.scaled(self.image_edge_draw.width(),
        #                            self.image_edge_draw.height())
        self.image_edge_draw.setPixmap(QPixmap.fromImage(showImage))

    #flood
    def update_flood_run(self):
        if self.img is None:
            return
        if len(self.lineEdit_img_w.text()):
            self.size_scale = float(self.lineEdit_img_w.text()) / self.src_w
        else:
            self.size_scale = 1
        h, w = self.img.shape[:2]
        draw_mask = np.zeros([h + 2, w + 2], np.uint8)

        area_info = ""
        draw_image = self.img.copy()
        if len(self.flood_point_list):
            for i in range(len(self.flood_point_list)):
                point = self.flood_point_list[i]
                if i==len(self.flood_point_list)-1:
                    fill_color =  (255, 0, 0)
                else:
                    fill_color = (255, 255, 0)
                copyimg = self.img.copy()

                h, w = copyimg.shape[:2]
                # 需要在原始图像像素长度上加2，否则算法会报错
                mask = np.zeros([h + 2, w + 2], np.uint8)

                retval, img1, flood_mask, rect = cv2.floodFill(copyimg, mask, (point[0], point[1]), fill_color,
                                                          (self.horizontalSlider_diff.value(), self.horizontalSlider_diff.value(), self.horizontalSlider_diff.value()),
                                                          (self.horizontalSlider_diff.value(), self.horizontalSlider_diff.value(), self.horizontalSlider_diff.value()),
                                                          cv2.FLOODFILL_FIXED_RANGE)

                draw_mask = cv2.bitwise_or(draw_mask,flood_mask)
                area = retval*self.size_scale*self.size_scale

                # 找出掩码轮廓点 RETR_EXTERNAL 只区最外层
                if version.startswith("3"):  # opencv 3版本，CHAIN_APPROX_SIMPLE保留所有轮廓上的点
                    _, contours, _ = cv2.findContours(flood_mask[1:flood_mask.shape[0]-1,1:flood_mask.shape[1]-1], cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)
                elif version.startswith("4"):  # opencv 4版本
                    contours, _ = cv2.findContours(flood_mask[1:flood_mask.shape[0]-1,1:flood_mask.shape[1]-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                else:
                    raise Exception("版本不支持:{}".format(version))

                (x_center, y_center), radius = cv2.minEnclosingCircle(contours[0])
                diameter = 2 * radius * self.size_scale
                # 轮廓点维度reshpe 成 n 行 2列，这样每行就是 一个x，y值
                point = np.reshape(contours[0], [-1, 2])
                # 在轮廓中心附近写上id
                cv2.drawContours(draw_image, contours, -1, fill_color, -1)
                cv2.putText(draw_image, str(i + 1), (int(x_center), int(y_center)), 1, self.src_w/150,
                            (255, 255, 255))
                self.area_line_point_info.append([area, diameter, point.tolist()])

                area_info += "id:{},area:{}\n".format(i+1, area)
        crop_draw_mask = draw_mask[1:draw_mask.shape[0]-1,1:draw_mask.shape[1]-1]*255
        self.textEdit_flood_info.setText(area_info)
        self.set_flood_mask(crop_draw_mask)
        self.set_flood_image_draw(draw_image)

    def flood_cancle(self):
        if len(self.flood_point_list) > 1:
            self.flood_point_list = self.flood_point_list[:-1]
        else:
            self.flood_point_list = []

        self.update_flood_run()

    def set_flood_mask(self, frame):
        showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1],
                         QImage.Format_Grayscale8)
        # showImage = showImage.scaled(self.image_flood_mask.width(),
        #                            self.image_flood_mask.height())
        self.image_flood_mask.setPixmap(QPixmap.fromImage(showImage))

    def set_flood_image_draw(self, frame):
        showImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.shape[1] * 3,
                         QImage.Format_RGB888).rgbSwapped()
        # showImage = showImage.scaled(self.image_flood_draw.width(),
        #                            self.image_flood_draw.height())
        self.image_flood_draw.setPixmap(QPixmap.fromImage(showImage))

    def _draw(self,data_list, increment, min_value, max_value,titile="", xlabel="", ylabel="frequency(%)"):
        data_list = sorted(data_list)
        if min_value =="":
            min_value = data_list[0]
        else:
            min_value = float(min_value)
        if max_value =="":
            max_value = data_list[-1]
        else:
            max_value = float(max_value)
        all_values_range = data_list[-1] - data_list[0] + 0.000001
        if increment !="":
            gap = float(increment)
            split_num = int(all_values_range / gap)
        else:  # 没有默认增量，则按照区间范围值的 1/10 来分
            split_num = 10
            gap = int(all_values_range / split_num)

        x_list = []
        y_list = []
        for i in range(0, split_num + 1):
            pre_split_list = []
            for data in data_list:
                if data >= min_value + gap * (i - 0.5) and data < min_value + gap * (i + 0.5):
                    pre_split_list.append(data)
            x_list.append(min_value + gap * i)
            # 区间占比
            y_list.append(len(pre_split_list) / len(data_list) * 100)

        plt.clf()

        # # 设置坐标轴刻度间隔
        x_major_locator = MultipleLocator(gap)
        # y_major_locator = MultipleLocator(2)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.xlim(min_value - 0.5 * gap, max_value + 0.5 * gap)
        # 占用宽度定为0.8
        plt.bar(x_list, y_list,width=int(gap * 0.8),color="b")
        plt.title(titile)
        plt.savefig(titile+".png")
        f = open(titile+".txt","w")
        for i in range(len(x_list)):
            f.write("{},{}\n".format(x_list[i],y_list[i]))
        f.close()

    def draw(self):
        if len(self.fig_area_list) > 1:
            self._draw(self.fig_area_list, self.lineEdit_fig_area_increment.text(),
                      self.lineEdit_fig_area_min_value.text(),
                      self.lineEdit_fig_area_max_value.text(), "grain area distribution")

            self._draw(self.fig_diameter_list, self.lineEdit_fig_line_increment.text(),
                      self.lineEdit_fig_line_min_value.text(),
                      self.lineEdit_fig_line_max_value.text(), "grain size distribution")

            area_png = QtGui.QPixmap("grain area distribution.png")
            self.label_fig_area.setPixmap(area_png)
            size_png = QtGui.QPixmap("grain size distribution.png")
            self.label_fig_line.setPixmap(size_png)


    #画图界面
    def open_data_file(self):
        # 读取本地图片开始执行
        file_path, _ = QFileDialog.getOpenFileName(self, 'open file', ".",
                                                    ("txt (*.txt)"))
        if file_path != "":
            row_values = open(file_path, "r").read().splitlines()
            self.fig_area_list = []
            self.fig_diameter_list = []
            for row in row_values:
                if len(row):
                    datas = row.split(" ")
                    self.fig_area_list.append(float(datas[0]))
                    self.fig_diameter_list.append(float(datas[1]))
            self.draw()

    def export_data(self):
        if len(self.area_line_point_info):
            open_file_name = None
            if self.tabWidget.currentIndex()==0:
                open_file_name = "hsv_data.txt"
            elif self.tabWidget.currentIndex()==1:
                open_file_name = "binary_data.txt"
            elif self.tabWidget.currentIndex() == 2:
                open_file_name = "edge_data.txt"
            elif self.tabWidget.currentIndex() == 3:
                open_file_name = "flood_data.txt"
            with open(open_file_name,"w") as f:
                for info in self.area_line_point_info:
                    f.write("{} {} {}\n".format(info[0],info[1],info[2]))
            QMessageBox.information(self, 'export data', "Export {} completed".format(open_file_name),
                                 QMessageBox.Ok)

        else:
            QMessageBox.critical(self, 'error', "No data needs to be exported",
                                 QMessageBox.Ok)

    def open_browser(self):
        self.browser = Browser()
        self.browser.show()


if __name__ == '__main__':
    # QtCore.QCoreApplication.setAttribute(
    #     QtCore.Qt.AA_EnableHighDpiScaling)  # 解决了Qtdesigner设计的界面与实际运行界面不一致的问题
    app = QApplication(sys.argv)
    w = mwindow()
    w.show()
    sys.exit(app.exec_())