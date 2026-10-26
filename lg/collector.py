"""Button-driven collection: freeze, label, save, return to live preview."""
import copy
import queue
import threading
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from .data import sample_counts, save_snapshot
from .vision import Hands, open_camera, draw
from .demo import scene, observations
from .ui import BG, CARD, TEXT, MUTED, ACCENT, BORDER
from .labels import LABELS, LOVE_LABEL, canonical_label,ACTIVE_DATASET


class CaptureSession:
    """Frozen observations cannot be replaced by later camera frames or saved twice."""
    def __init__(self,path,operator='p01',session=None):
        self.path=Path(path)
        self.operator=operator
        self.session=session or uuid.uuid4().hex[:10]
        self.frozen=None
        self.counts=sample_counts(path)
        self.saved=0

    def freeze(self,hands,operator,side='both'):
        if self.frozen is not None:
            raise ValueError('请先保存或取消当前采集')
        operator=operator.strip()
        if not operator:
            raise ValueError('请填写操作员编号，例如 p01')
        selected=[h for h in hands if side=='both' or ('left' if h.points[0,0]<.5 else 'right')==side]
        if len(selected)!=1:
            raise ValueError(f'请只放一只手进入采集区域；当前检测到 {len(selected)} 只')
        if operator!=self.operator:
            self.operator=operator
            self.session=uuid.uuid4().hex[:10]
        self.frozen=copy.deepcopy(selected)

    def save(self,label,hand):
        if self.frozen is None:
            raise ValueError('没有待保存的画面，请先点击采集')
        if hand not in ('left','right'):
            raise ValueError('请选择实际左手或右手')
        label=canonical_label(label)
        row=save_snapshot(self.path,self.frozen,label,self.operator,self.session,hand,capture='gui_snapshot')
        self.counts[(hand,label)]+=1
        self.saved+=1
        self.frozen=None
        return row

    def cancel(self):
        self.frozen=None

    def new_session(self,operator):
        if self.frozen is not None:
            raise ValueError('请先保存或取消当前采集')
        if not operator.strip():
            raise ValueError('请填写操作员编号')
        self.operator=operator.strip()
        self.session=uuid.uuid4().hex[:10]


class CollectorApp:
    def __init__(self,args,root_path):
        self.args=args
        self.root_path=Path(root_path)
        path=self.root_path/('outputs/collection_demo/digits.jsonl' if args.demo else ACTIVE_DATASET)
        self.capture=CaptureSession(path,args.operator,args.session)
        self.root=tk.Tk()
        self.root.title('YTO手势数据采集工具'+(' | 离线模拟' if args.demo else ''))
        self.root.geometry(f'1180x{min(920,self.root.winfo_screenheight()-100)}')
        self.root.minsize(1000,720)
        self.root.configure(bg=BG)
        self.stop=threading.Event()
        self.frames=queue.Queue(maxsize=2)
        self.latest=None
        self.dialog=None
        self.label=canonical_label(args.label)
        self.hand=args.hand
        self.shown=0
        self.make_ui()
        self.worker=threading.Thread(target=self.run_worker,daemon=True)
        self.worker.start()
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.after(35,self.poll)
        if args.smoke:
            self.root.after(int(args.smoke*1000),self.close)

    def make_ui(self):
        font='Microsoft YaHei UI'
        style=ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('Capture.TButton',font=(font,12,'bold'),padding=14,
                        background=ACCENT,foreground='white',borderwidth=0)
        style.map('Capture.TButton',background=[('active','#6036c5'),('disabled','#cac1df')])
        style.configure('Secondary.TButton',font=(font,10),padding=9,background='#eee9f8',foreground=ACCENT,borderwidth=0)
        header=tk.Frame(self.root,bg=CARD,padx=22,pady=15)
        header.pack(fill='x')
        tk.Label(header,text='YTO  /  手势数据采集',bg=CARD,fg=ACCENT,font=(font,20,'bold')).pack(anchor='w')
        tk.Label(header,text='摆好手势 → 点击采集 → 选择数字与左右手 → 保存 → 下一轮',bg=CARD,fg=MUTED,font=(font,11)).pack(anchor='w',pady=(6,0))
        body=tk.Frame(self.root,bg=BG,padx=18,pady=16)
        body.pack(fill='both',expand=True)
        body.columnconfigure(0,weight=1); body.columnconfigure(1,minsize=320)
        body.rowconfigure(0,weight=1)
        left=tk.Frame(body,bg=CARD,padx=10,pady=10)
        left.grid(row=0,column=0,sticky='nsew',padx=(0,16))
        self.input_status=tk.Label(left,text='正在连接摄像头…',bg=CARD,fg=MUTED,font=(font,10),anchor='w')
        self.input_status.pack(fill='x',padx=5,pady=(3,10))
        self.preview=tk.Label(left,bg='#121522')
        self.preview.pack(fill='both',expand=True)
        tk.Label(left,text='手背朝摄像头，每次只采一只手；另一只手操作鼠标。',bg=CARD,fg=MUTED,font=(font,10),pady=12).pack()
        right=tk.Frame(body,bg=CARD,padx=18,pady=16)
        right.grid(row=0,column=1,sticky='nsew')
        tk.Label(right,text='采集设置',bg=CARD,fg=TEXT,font=(font,13,'bold')).pack(anchor='w')
        tk.Label(right,text='操作员编号',bg=CARD,fg=MUTED,font=(font,10)).pack(anchor='w',pady=(16,6))
        self.operator=tk.StringVar(value=self.args.operator)
        ttk.Entry(right,textvariable=self.operator,font=(font,12),width=23).pack(fill='x')
        self.session_label=tk.Label(right,text='',bg=CARD,fg=MUTED,font=(font,9),anchor='w')
        self.session_label.pack(fill='x',pady=10)
        self.capture_button=ttk.Button(right,text='采集当前手势',style='Capture.TButton',command=self.open_capture,state='disabled')
        self.capture_button.pack(fill='x',pady=(3,10))
        ttk.Button(right,text='开始新的独立采集轮次',style='Secondary.TButton',command=self.new_session).pack(fill='x')
        self.feedback=tk.Label(right,text='摆好手势后，点击“采集当前手势”',bg='#f3eefb',fg=ACCENT,font=(font,10),
                               wraplength=270,justify='left',anchor='nw',padx=10,pady=8,height=3)
        self.feedback.pack(fill='x',pady=10)
        tk.Label(right,text='已保存数量 · 左右手分别统计',bg=CARD,fg=TEXT,font=(font,11,'bold')).pack(anchor='w',pady=(0,7))
        table_area=tk.Frame(right,bg=CARD)
        table_area.pack(fill='both',expand=True)
        self.table=ttk.Treeview(table_area,columns=('digit','left','right'),show='headings',height=11)
        for name,label,width in [('digit','数字',90),('left','左手',75),('right','右手',75)]:
            self.table.heading(name,text=label); self.table.column(name,width=width,anchor='center',stretch=True)
        self.table.pack(side='left',fill='both',expand=True)
        scrollbar=ttk.Scrollbar(table_area,command=self.table.yview)
        scrollbar.pack(side='right',fill='y')
        self.table.config(yscrollcommand=scrollbar.set)
        tk.Label(right,text='每次保存 = 1 个关键点样本\n建议每手每类每轮 30–50 个，至少 3 轮\n5–6 个仅用于验证采集流程',bg=CARD,fg=MUTED,
                 justify='left',font=(font,9)).pack(anchor='w',pady=12)
        self.refresh_counts()

    def refresh_counts(self):
        self.session_label.config(text=f'当前轮次：{self.capture.session}  |  本次已存 {self.capture.saved}')
        for item in self.table.get_children(): self.table.delete(item)
        for label in LABELS:
            self.table.insert('','end',values=(label,self.capture.counts[('left',label)],self.capture.counts[('right',label)]))

    def run_worker(self):
        cap=detector=None
        try:
            if self.args.demo:
                base=scene()
            else:
                detector=Hands(self.root_path,classify=False)
                cap,base,_=open_camera(self.args.camera)
            while not self.stop.is_set():
                hands=observations()[:1] if self.args.demo else detector.process(base)
                packet=dict(frame=base.copy(),hands=hands)
                if self.frames.full():
                    try: self.frames.get_nowait()
                    except queue.Empty: pass
                self.frames.put_nowait(packet)
                if cap:
                    ok,base=cap.read()
                    if not ok: raise RuntimeError('摄像头读取失败或视频结束')
                else: self.stop.wait(.04)
        except Exception as e:
            if self.frames.full():
                try: self.frames.get_nowait()
                except queue.Empty: pass
            self.frames.put_nowait(dict(error=f'{type(e).__name__}: {e}'))
        finally:
            if cap: cap.release()
            if detector: detector.close()

    def poll(self):
        packet=None
        while not self.frames.empty(): packet=self.frames.get_nowait()
        if packet:
            if 'error' in packet:
                self.latest=None
                self.capture_button.config(state='disabled')
                self.input_status.config(text=packet['error'],fg='#c83d55',wraplength=650)
                self.feedback.config(text=packet['error'],fg='#c83d55')
            else:
                self.latest=packet
                if self.dialog is None:
                    self.show_frame(self.preview,draw(packet['frame'],packet['hands']))
                    self.capture_button.config(state='normal')
                    self.input_status.config(text=f"{'离线模拟（数据单独保存）' if self.args.demo else '实时预览 · 摄像头 '+self.args.camera}  |  检测到 {len(packet['hands'])} 只手")
                    self.shown+=1
        if not self.stop.is_set(): self.root.after(40,self.poll)

    def show_frame(self,widget,frame,max_size=None):
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        image.thumbnail(max_size or (max(1,widget.winfo_width()),max(1,widget.winfo_height())))
        widget.photo=ImageTk.PhotoImage(image)
        widget.config(image=widget.photo)

    def open_capture(self):
        if self.dialog is not None or not self.latest: return
        try:
            self.capture.freeze(self.latest['hands'],self.operator.get(),self.args.side)
        except ValueError as e:
            self.feedback.config(text=str(e),fg='#c83d55'); return
        self.frozen_frame=self.latest['frame'].copy()
        self.capture_button.config(state='disabled')
        self.input_status.config(text='画面已冻结，保存的是点击采集时的姿态')
        dialog=tk.Toplevel(self.root)
        self.dialog=dialog
        dialog.title('选择手势标签并保存')
        dialog.configure(bg=CARD)
        dialog.resizable(False,False)
        dialog.transient(self.root)
        dialog.protocol('WM_DELETE_WINDOW',self.cancel_capture)
        wrap=tk.Frame(dialog,bg=CARD,padx=22,pady=18)
        wrap.pack(fill='both',expand=True)
        tk.Label(wrap,text='给当前画面标注',bg=CARD,fg=TEXT,font=('Microsoft YaHei UI',16,'bold')).pack(anchor='w')
        tk.Label(wrap,text=f'操作员 {self.capture.operator} · 轮次 {self.capture.session}',bg=CARD,fg=MUTED,font=('Microsoft YaHei UI',9)).pack(anchor='w',pady=(4,10))
        preview=tk.Label(wrap,bg='#121522')
        preview.pack(fill='x')
        self.show_frame(preview,draw(self.frozen_frame,self.capture.frozen),max_size=(440,220))
        self.label_var=tk.StringVar(value=self.label)
        self.hand_var=tk.StringVar(value=self.hand or '')
        tk.Label(wrap,text='1. 选择手势（9 = 握拳）',bg=CARD,fg=TEXT,font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w',pady=(14,6))
        digits=tk.Frame(wrap,bg=CARD); digits.pack(fill='x')
        self.label_buttons={}
        for i,label in enumerate(LABELS):
            button=tk.Button(digits,text=label,
                command=lambda v=label:self.choose_label(v),relief='flat',bd=0,pady=8,
                font=('Microsoft YaHei UI',11),cursor='hand2')
            row,column,span=(2,0,3) if label==LOVE_LABEL else ((2,3,2) if label=='ok' else (i//5,i%5,1))
            button.grid(row=row,column=column,columnspan=span,sticky='ew',padx=3,pady=3)
            digits.columnconfigure(i%5,weight=1)
            self.label_buttons[label]=button
        tk.Label(wrap,text='2. 选择人的实际左手 / 右手',bg=CARD,fg=TEXT,font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w',pady=(12,6))
        hand_row=tk.Frame(wrap,bg=CARD); hand_row.pack(fill='x')
        self.hand_buttons={}
        for i,(value,label) in enumerate([('left','左手'),('right','右手')]):
            button=tk.Button(hand_row,text=label,command=lambda v=value:self.choose_hand(v),
                             relief='flat',bd=0,pady=10,font=('Microsoft YaHei UI',11),cursor='hand2')
            button.grid(row=0,column=i,sticky='ew',padx=3); hand_row.columnconfigure(i,weight=1)
            self.hand_buttons[value]=button
        self.dialog_feedback=tk.Label(wrap,text='请确认数字和实际左右手，保存后自动继续下一轮',bg=CARD,fg=MUTED,
                                      wraplength=440,font=('Microsoft YaHei UI',9))
        self.dialog_feedback.pack(pady=12)
        ttk.Button(wrap,text='保存样本，继续采集',style='Capture.TButton',command=self.save_capture).pack(fill='x')
        ttk.Button(wrap,text='取消，重新摆手势',style='Secondary.TButton',command=self.cancel_capture).pack(fill='x',pady=(8,0))
        self.update_choices()
        dialog.update_idletasks()
        x=self.root.winfo_rootx()+(self.root.winfo_width()-dialog.winfo_reqwidth())//2
        y=self.root.winfo_rooty()+max(0,(self.root.winfo_height()-dialog.winfo_reqheight())//2)
        dialog.geometry(f'+{max(0,x)}+{max(0,y)}')
        dialog.grab_set()

    def choose_label(self,label):
        self.label_var.set(label); self.update_choices()

    def choose_hand(self,hand):
        self.hand_var.set(hand); self.update_choices()

    def update_choices(self):
        for group,value in [(self.label_buttons,self.label_var.get()),(self.hand_buttons,self.hand_var.get())]:
            for label,button in group.items():
                button.config(bg=ACCENT if label==value else '#f1eef8',fg='white' if label==value else TEXT)

    def save_capture(self):
        try:
            row=self.capture.save(self.label_var.get(),self.hand_var.get())
        except (ValueError,OSError) as e:
            self.dialog_feedback.config(text=str(e),fg='#c83d55'); return
        self.label,self.hand=row['label'],row['physical_hand']
        name='左手' if self.hand=='left' else '右手'
        self.feedback.config(text=f'保存成功 +1\n{name} / {self.label}：累计 {self.capture.counts[(self.hand,self.label)]} 个\n请调整姿态，继续点击采集',fg='#21866e')
        self.finish_dialog()
        self.refresh_counts()

    def cancel_capture(self):
        self.capture.cancel()
        self.finish_dialog()

    def finish_dialog(self):
        if self.dialog:
            self.dialog.grab_release(); self.dialog.destroy(); self.dialog=None
        self.latest=None  # require a fresh camera observation before the next capture

    def new_session(self):
        try: self.capture.new_session(self.operator.get())
        except ValueError as e:
            self.feedback.config(text=str(e),fg='#c83d55'); return
        self.refresh_counts()
        self.feedback.config(text='已开始新的独立轮次\n请改变位置、角度或光照后采集\n每轮应覆盖相同的数字和左右手',fg=ACCENT)

    def close(self):
        self.stop.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        self.worker.join(timeout=5)
        if self.args.smoke:
            print(f'COLLECTOR_SMOKE rendered_frames={self.shown}')
            if not self.shown: raise RuntimeError('No collection preview rendered')
