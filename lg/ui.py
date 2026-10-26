"""YTO desktop presentation; camera and business logic live in their own modules."""
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image,ImageTk
from .controls import OPERATOR_GUIDE,GATE_IDS

BG='#f3f4f9'
CARD='#ffffff'
TEXT='#202035'
MUTED='#7b7f95'
ACCENT='#7044d6'
BORDER='#e7e8f0'


def build_ui(self, modes):
    font='Microsoft YaHei UI'
    style=ttk.Style(self.root)
    style.theme_use('clam')
    style.configure('YTO.TButton',font=(font,10),padding=(10,8),background='#f1edfb',
                    foreground=ACCENT,borderwidth=0,focusthickness=0)
    style.map('YTO.TButton',background=[('active','#e5dcfa'),('pressed','#d9cbf5')])
    style.configure('Compact.TButton',font=(font,9),padding=(8,4),background='#f1edfb',
                    foreground=ACCENT,borderwidth=0,focusthickness=0)
    style.map('Compact.TButton',background=[('active','#e5dcfa')])
    style.configure('Danger.TButton',font=(font,11,'bold'),padding=10,background='#fff0f1',
                    foreground='#cf4359',borderwidth=0,focusthickness=0)
    style.map('Danger.TButton',background=[('active','#ffdce2'),('pressed','#ffc6d0')])
    style.configure('YTO.Horizontal.TProgressbar',background=ACCENT,troughcolor='#eeebf6',
                    borderwidth=0,lightcolor=ACCENT,darkcolor=ACCENT,thickness=6)
    self.root.configure(bg=BG)
    header=tk.Frame(self.root,bg=CARD,height=92)
    header.pack(fill='x')
    header.pack_propagate(False)
    logo_path=self.root_path/'docs/assets/yto-logo.jpg'
    if logo_path.is_file():
        with Image.open(logo_path) as source:
            mark=source.convert('RGB');mark.thumbnail((112,68))
        self.brand_logo=ImageTk.PhotoImage(mark)
        logo=tk.Label(header,image=self.brand_logo,bg=CARD,bd=0)
    else:
        logo=tk.Label(header,text='YTO',bg=ACCENT,fg='white',font=('Segoe UI',23,'bold'),padx=15,pady=5)
    logo.pack(side='left',padx=(24,15),pady=17)
    names=tk.Frame(header,bg=CARD)
    names.pack(side='left')
    tk.Label(names,text='YTO手势隔空操作系统',bg=CARD,fg=TEXT,font=(font,19,'bold')).pack(anchor='w')
    tk.Label(names,text='智慧物流工作站   /   GESTURE CONTROL',bg=CARD,fg=MUTED,font=(font,9)).pack(anchor='w',pady=(4,0))
    self.connection=tk.Label(header,text='●  离线演示' if self.args.demo else '●  正在连接',
                             bg='#f0ebfb',fg=ACCENT,font=(font,10,'bold'),padx=14,pady=7)
    self.connection.pack(side='right',padx=24)
    self.status_panel=tk.Frame(self.root,bg=CARD,highlightbackground=BORDER,highlightthickness=1,padx=18,pady=10)
    self.status_panel.pack(fill='x',padx=22,pady=(12,0))
    state_row=tk.Frame(self.status_panel,bg=CARD);state_row.pack(fill='x')
    self.belt=tk.Label(state_row,text='●  传送带初始化中',bg=CARD,fg=TEXT,font=(font,25,'bold'),anchor='w')
    self.belt.pack(side='left')
    state_meta=tk.Frame(state_row,bg=CARD);state_meta.pack(side='right')
    self.gate_status=tk.Label(state_meta,text='格口未绑定',bg='#f1edfb',fg=MUTED,
                              font=(font,25,'bold'),padx=16,pady=6)
    self.gate_status.pack(anchor='e')
    tk.Label(state_meta,text='状态来源：本地模拟 · 未连接 PLC 反馈',bg=CARD,fg=MUTED,font=(font,8)).pack(anchor='e')
    self.command_status=tk.Label(self.status_panel,text='最近指令：等待操作',bg=CARD,fg=TEXT,font=(font,11),anchor='w')
    self.command_status.pack(fill='x',pady=(3,8))
    guide=tk.Frame(self.status_panel,bg=CARD);guide.pack(fill='x')
    for column,(name,method) in enumerate(OPERATOR_GUIDE):
        guide.columnconfigure(column,weight=1,uniform='guide')
        tile=tk.Frame(guide,bg='#f5f5fa',padx=10,pady=7)
        tile.grid(row=0,column=column,sticky='nsew',padx=(0,8) if column<2 else 0)
        tk.Label(tile,text=name,bg='#f5f5fa',fg=ACCENT,font=(font,10,'bold')).pack(anchor='w')
        tk.Label(tile,text=method,bg='#f5f5fa',fg=TEXT,font=(font,9),wraplength=310,justify='left').pack(anchor='w',pady=(2,0))
    footer=tk.Frame(self.root,bg=BG)
    footer.pack(side='bottom',fill='x',padx=24,pady=(2,10))
    tk.Label(footer,text='格口仅支持 2–5 · 同数绑定请进入“格口绑定”模式',
             bg=BG,fg=MUTED,font=(font,9)).pack(side='left')
    tk.Label(footer,text='本地模拟控制 · 未连接 PLC',bg=BG,fg=MUTED,font=(font,9)).pack(side='right')
    body=tk.Frame(self.root,bg=BG)
    body.pack(fill='both',expand=True,padx=22,pady=(18,0))
    body.columnconfigure(0,weight=1)
    body.columnconfigure(1,minsize=350)
    body.rowconfigure(0,weight=1)
    left=tk.Frame(body,bg=CARD,highlightbackground=BORDER,highlightthickness=1)
    left.grid(row=0,column=0,sticky='nsew',padx=(0,18))
    preview_header=tk.Frame(left,bg=CARD)
    preview_header.pack(fill='x',padx=18,pady=(16,10))
    tk.Label(preview_header,text='实时视觉',bg=CARD,fg=TEXT,font=(font,13,'bold')).pack(side='left')
    tk.Label(preview_header,text=f'双指锁定  {self.args.dwell:g} s',bg='#f2edfb',fg=ACCENT,
             font=(font,9),padx=10,pady=4).pack(side='right')
    self.camera_info=tk.Label(left,text='正在初始化摄像头与识别模块…',bg=CARD,fg=MUTED,
                              font=(font,9),anchor='w')
    self.camera_info.pack(fill='x',padx=18,pady=(0,10))
    self.preview=tk.Label(left,bg='#121522',bd=0)
    self.preview.pack(fill='both',expand=True,padx=12,pady=0)
    self.hint=tk.Label(left,text=f'两食指指向目标对角 · 保持 {self.args.dwell:g} 秒 · 圆圈闭合即选中',
                       bg=CARD,fg=MUTED,font=(font,10),pady=15)
    self.hint.pack(fill='x',padx=12)
    sidebar=tk.Frame(body,bg=BG,width=370)
    sidebar.grid(row=0,column=1,sticky='nsew')
    canvas=tk.Canvas(sidebar,bg=BG,highlightthickness=0,width=350)
    side_scroll=ttk.Scrollbar(sidebar,orient='vertical',command=canvas.yview)
    side_scroll.pack(side='right',fill='y'); canvas.pack(side='left',fill='both',expand=True)
    canvas.configure(yscrollcommand=side_scroll.set)
    right=tk.Frame(canvas,bg=BG,width=350)
    window=canvas.create_window((0,0),window=right,anchor='nw')
    right.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width))
    def wheel(event):
        if str(event.widget).startswith(str(sidebar)):
            canvas.yview_scroll(-int(event.delta/120),'units')
    self.root.bind_all('<MouseWheel>',wheel,add='+')
    right.columnconfigure(0,weight=1)
    log_row=4 if self.args.demo else 3
    right.rowconfigure(log_row,weight=1)

    def card(row):
        f=tk.Frame(right,bg=CARD,highlightbackground=BORDER,highlightthickness=1,padx=14,pady=10)
        f.grid(row=row,column=0,sticky='nsew',pady=(0,12))
        return f

    def title(parent,text):
        tk.Label(parent,text=text,bg=CARD,fg=TEXT,font=(font,11,'bold')).pack(anchor='w',pady=(0,6))

    operation=card(0)
    title(operation,'操作模式')
    buttons=tk.Frame(operation,bg=CARD)
    buttons.pack(fill='x')
    self.mode_var=tk.StringVar(value=next(k for k,v in modes.items() if v==self.args.mode))
    self.mode_buttons={}
    def select_mode(name):
        self.mode_var.set(name)
        self.commands.put(('mode',modes[name]))
        for label,button in self.mode_buttons.items():
            active=label==name
            button.config(bg=ACCENT if active else '#f5f5fa',fg='white' if active else MUTED,
                          activebackground='#6236c4' if active else '#eeebf6')
    for i,name in enumerate(modes):
        active=self.mode_var.get()==name
        button=tk.Button(buttons,text=name,command=lambda n=name:select_mode(n),
                         bg=ACCENT if active else '#f5f5fa',fg='white' if active else MUTED,
                         activebackground='#6236c4',activeforeground='white',font=(font,10),
                         relief='flat',bd=0,cursor='hand2',pady=7)
        last_wide=i==len(modes)-1 and len(modes)%2==1
        button.grid(row=i//2,column=i%2,columnspan=2 if last_wide else 1,sticky='ew',padx=(0,6) if i%2==0 and not last_wide else (0,0),pady=3)
        buttons.columnconfigure(i%2,weight=1)
        self.mode_buttons[name]=button
    status=card(1)
    title(status,'业务结果与识别进度')
    reading=tk.Frame(status,bg=CARD)
    reading.pack(fill='x')
    self.gesture=tk.Label(reading,text='等待双手',bg=CARD,fg=TEXT,font=(font,11))
    self.gesture.pack(side='left')
    self.hold_status=tk.Label(reading,text='等待锁定',bg=CARD,fg=ACCENT,font=(font,10))
    self.hold_status.pack(side='right')
    self.progress=ttk.Progressbar(status,maximum=1,style='YTO.Horizontal.TProgressbar')
    self.progress.pack(fill='x',pady=(7,11))
    self.count=tk.Label(status,text='唯一条码  0     待确认异常  0',bg=CARD,fg=MUTED,font=(font,9))
    self.count.pack(anchor='w',pady=(7,3))
    self.action_status=tk.Label(status,text='',bg=CARD,fg=ACCENT,font=(font,9),wraplength=306,justify='left')
    self.action_status.pack(fill='x')
    self.action_capture=tk.Frame(status,bg=CARD)
    self.action_label=tk.StringVar(value='pinch')
    ttk.Combobox(self.action_capture,textvariable=self.action_label,values=('pinch','fist','idle'),state='readonly',width=8).pack(side='left')
    ttk.Button(self.action_capture,text='录制 2 秒动作样本',style='Compact.TButton',
               command=lambda:self.commands.put(('record_action',self.action_label.get()))).pack(side='right')
    self.result=tk.Label(status,text='识别结果将在此显示',bg=CARD,fg=TEXT,font=(font,10),
                         wraplength=306,justify='left',anchor='nw',height=2)
    self.result.pack(fill='x',pady=(5,0))
    self.preview_title=tk.Label(status,text='最近扫码区域',bg=CARD,fg=MUTED,font=(font,8))
    self.preview_title.pack(anchor='w',pady=(5,3))
    scan_area=tk.Frame(status,bg='#f4f4f8',height=90)
    scan_area.pack(fill='x');scan_area.pack_propagate(False)
    self.scan_preview=tk.Label(scan_area,bg='#f4f4f8',text='等待框选',fg=MUTED,font=(font,9))
    self.scan_preview.pack(fill='both',expand=True)
    actions=card(2)
    ttk.Button(actions,text='启动传送带（模拟）',style='YTO.TButton',
               command=lambda:self.commands.put(('start',None))).pack(fill='x',pady=(0,8))
    gate_row=tk.Frame(actions,bg=CARD);gate_row.pack(fill='x',pady=(0,8))
    self.gate_choice=tk.StringVar(value='2')
    ttk.Combobox(gate_row,textvariable=self.gate_choice,values=GATE_IDS,state='readonly',width=4).pack(side='left')
    ttk.Button(gate_row,text='绑定摆臂格口',style='Compact.TButton',
               command=lambda:self.commands.put(('bind_gate',int(self.gate_choice.get())))).pack(side='right')
    ttk.Button(actions,text='人工复位急停',style='YTO.TButton',
               command=lambda:self.commands.put(('reset',None))).pack(fill='x')
    ttk.Button(actions,text='停止并锁定  ·  模拟急停',style='Danger.TButton',
               command=lambda:self.commands.put(('estop',None))).pack(fill='x',pady=(9,0))
    if self.args.demo:
        demo=card(3)
        title(demo,'模拟输入 · 绕过视觉模型')
        grid=tk.Frame(demo,bg=CARD)
        grid.pack(fill='x')
        for i,(label,value) in enumerate([('双指框选',(1,1)),('双手 OK（需两次）',('ok','ok')),('握拳保持 1 秒',(9,9)),('张开 / 释放',(5,5)),('移开双手',None),('双手3：格口模式',(3,3))]):
            ttk.Button(grid,text=label,style='Compact.TButton',command=lambda d=value:self.commands.put(('demo',d))).grid(
                row=i//2,column=i%2,sticky='ew',padx=(0,6) if i%2==0 else 0,pady=2)
            grid.columnconfigure(i%2,weight=1)
        for i,(label,value) in enumerate([('动作：展开','open'),('动作：捏合','pinch'),('动作：握拳','fist')]):
            ttk.Button(grid,text=label,style='Compact.TButton',command=lambda d=value:self.commands.put(('demo_action',d))).grid(
                row=3+i//2,column=i%2,sticky='ew',pady=2)
    logs=card(log_row)
    title(logs,'操作记录')
    self.log=tk.Text(logs,height=3,width=1,bg=CARD,fg=MUTED,font=(font,9),wrap='word',
                     state='disabled',relief='flat',bd=0,highlightthickness=0,spacing3=6)
    scrollbar=ttk.Scrollbar(logs,command=self.log.yview)
    scrollbar.pack(side='right',fill='y')
    self.log.config(yscrollcommand=scrollbar.set)
    self.log.pack(fill='both',expand=True)


def render_packet(self,packet):
    if 'error' in packet:
        self.camera_info.config(text=packet['error'],fg='#ff9b9b',wraplength=790)
        self.result.config(text=packet['error'])
        self.belt.config(text='传送带：停止 / 输入错误',fg='#ff8080')
        self.gesture.config(text='输入不可用')
        self.command_status.config(text='最近指令：输入异常，传送带已停止')
        self.connection.config(text='●  输入异常',fg='#cf4359',bg='#fff0f1')
        self.hold_status.config(text='输入不可用',fg='#cf4359')
        self.progress['value']=0
    else:
        self.shown+=1
        self.camera_info.config(text=packet['info'])
        self.connection.config(text='●  离线演示' if self.args.demo else '●  实时识别中',
                               fg='#7044d6' if self.args.demo else '#21866e',bg='#f0ebfb' if self.args.demo else '#eaf7f2')
        image=Image.fromarray(cv2.cvtColor(packet['frame'],cv2.COLOR_BGR2RGB))
        image.thumbnail((max(1,self.preview.winfo_width()),max(1,self.preview.winfo_height())))
        self.photo=ImageTk.PhotoImage(image)
        self.preview.config(image=self.photo)
        self.gesture.config(text='当前识别：'+packet['digits'])
        self.action_status.config(text=packet.get('action_status',''))
        if packet.get('action_mode'):
            self.action_capture.pack(fill='x',pady=(8,0))
            self.hint.config(text='先展开手 → 捏合/握拳 → 再展开；完成一轮只输出一次')
        else:
            self.action_capture.pack_forget()
            self.hint.config(text=f'两食指指向目标对角 · 保持 {self.args.dwell:g} 秒 · 圆圈闭合即选中')
        fraction=packet.get('control_progress',0) if packet.get('control_name') else packet['progress']
        self.progress['value']=fraction
        self.hold_status.config(text='已锁定' if fraction>=1 else (f'锁定中 {fraction:.0%}' if fraction>0 else '等待锁定'),fg='#7044d6')
        if packet.get('scanning'): self.hold_status.config(text='正在扫码…')
        if packet.get('action_mode'): self.hold_status.config(text='录制中…' if packet.get('recording') else '时序识别')
        if packet.get('gate_mode'):
            self.hint.config(text='双手比出相同数字 2–5 · 稳定保持后绑定对应摆臂格口')
        if packet.get('control_name'):
            label=packet['control_name']
            self.hold_status.config(text=f'{label} {fraction:.0%}' if fraction<1 else f'{label}已提交')
        if packet.get('damage_mode'):
            self.hint.config(text='YOLO 后台连续检测 · 双指锁定显示结果 · 移开手指隐藏')
        mode=packet.get('preview_mode','scan')
        self.preview_title.config(text={'photo':'最新拍照 · 保留至下一张','damage':'当前框选识别结果 · 移开手指隐藏'}.get(mode,'最近扫码区域'))
        if getattr(self,'preview_mode',None)!=mode:
            self.scan_preview.config(image='',text='等待框选拍照' if mode=='photo' else '等待框选')
            self.shown_scan_revision=None;self.preview_mode=mode
        if getattr(self,'scan_preview',None) is not None and packet.get('clear_preview'):
            self.scan_preview.config(image='',text='移开手指 · 识别结果已隐藏')
            self.shown_scan_revision=-1
        if getattr(self,'scan_preview',None) is not None and packet.get('scan_image') is not None:
            if getattr(self,'shown_scan_revision',-1)!=packet['scan_revision']:
                scan_image=Image.fromarray(cv2.cvtColor(packet['scan_image'],cv2.COLOR_BGR2RGB))
                scan_image.thumbnail((300,86))
                self.scan_photo=ImageTk.PhotoImage(scan_image)
                self.scan_preview.config(image=self.scan_photo)
                self.shown_scan_revision=packet['scan_revision']
        self.belt.config(text='●  传送带急停锁定' if packet['estop'] else ('●  传送带运行中' if packet['running'] else '●  传送带已停止'),
                         fg='#cf4359' if packet['estop'] else ('#21866e' if packet['running'] else '#c24153'),
                         bg=CARD)
        self.command_status.config(text='最近指令：'+packet.get('last_command','等待操作'))
        gate=packet.get('bound_gate')
        self.gate_status.config(text=f'已绑定：{gate} 号格口' if gate else '格口未绑定',
                                fg=ACCENT if gate else MUTED)
        self.count.config(text=f"唯一条码  {packet['count']}     待确认异常  {packet['pending']}")
        events=packet['events']
        if events and self.last_event!=(len(events),events[-1]['time']):
            self.last_event=(len(events),events[-1]['time'])
            self.result.config(text=packet.get('last_command','等待操作') if events[-1]['kind']=='action' else events[-1]['message'])
            self.log.config(state='normal')
            self.log.delete('1.0','end')
            self.log.insert('end','\n'.join(e['message'] for e in events[-30:]))
            self.log.see('end'); self.log.config(state='disabled')
        if packet.get('damage_message') is not None: self.result.config(text=packet['damage_message'])
