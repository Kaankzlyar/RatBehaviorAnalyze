function varargout = TMaze_v1(varargin)
% TMAZE_V1 MATLAB code for TMaze_v1.fig
%      TMAZE_V1, by itself, creates a new TMAZE_V1 or raises the existing
%      singleton*.
%
%      H = TMAZE_V1 returns the handle to a new TMAZE_V1 or the handle to
%      the existing singleton*.
%
%      TMAZE_V1('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TMAZE_V1.M with the given input arguments.
%
%      TMAZE_V1('Property','Value',...) creates a new TMAZE_V1 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before TMaze_v1_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to TMaze_v1_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help TMaze_v1

% Last Modified by GUIDE v2.5 12-Feb-2019 13:55:06

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @TMaze_v1_OpeningFcn, ...
                   'gui_OutputFcn',  @TMaze_v1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before TMaze_v1 is made visible.
function TMaze_v1_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to TMaze_v1 (see VARARGIN)
handles.output = hObject;
path(path,'TMazeSources')
axes(handles.axes1);
imshow(imread('BackGround.jpg'));
axis normal;

str = handles.edit1.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt1 = [x1, y1];
str = handles.edit2.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt2 = [x1, y1];
str = handles.edit3.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt3 = [x1, y1];
str = handles.edit4.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt4 = [x1, y1];
str = handles.edit5.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt5 = [x1, y1];
str = handles.edit6.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt6 = [x1, y1];
str = handles.edit7.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt7 = [x1, y1];
str = handles.edit8.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt8 = [x1, y1];
str = handles.edit9.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt9 = [x1, y1];
str = handles.edit10.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt10 = [x1, y1];
str = handles.edit11.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt11 = [x1, y1];
str = handles.edit12.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
handles.pnt12 = [x1, y1];

handles.axes2.Visible = 'off';
handles.axes3.Visible = 'off';
handles.axes4.Visible = 'off';
handles.axes5.Visible = 'off';
handles.pushbutton2.Visible = 'off';
handles.pushbutton3.Visible = 'off';
handles.slider2.Visible = 'off';
handles.text2.Visible = 'off';
handles.text3.Visible = 'off';
handles.text4.Visible = 'off';
handles.edit1.Visible = 'off';
handles.edit2.Visible = 'off';
handles.edit3.Visible = 'off';
handles.edit4.Visible = 'off';
handles.edit5.Visible = 'off';
handles.edit6.Visible = 'off';
handles.edit7.Visible = 'off';
handles.edit8.Visible = 'off';
handles.edit9.Visible = 'off';
handles.edit10.Visible = 'off';
handles.edit11.Visible = 'off';
handles.edit12.Visible = 'off';
handles.edit13.Visible = 'off';
handles.edit14.Visible = 'off';
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes TMaze_v1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = TMaze_v1_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit1.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt1 = [x1, y1];
handles.edit1.String = sprintf('%.0f, %0.f',x1,y1);

% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit2.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt2 = [x1, y1];
handles.edit2.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit3.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt3 = [x1, y1];
handles.edit3.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit4.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt4 = [x1, y1];
handles.edit4.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit5.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt5 = [x1, y1];
handles.edit5.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit6.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt6 = [x1, y1];
handles.edit6.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit7.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt7 = [x1, y1];
handles.edit7.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit8.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt8 = [x1, y1];
handles.edit8.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit9.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt9 = [x1, y1];
handles.edit9.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit10.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt10 = [x1, y1];
handles.edit10.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit11.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt11 = [x1, y1];
handles.edit11.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit12.String;
x1 = round(str2double(str(1:find(str==',',1))));
y1 = round(str2double(str(find(str==',',1)+1:end)));
if y1 < 1; y1 = 1; end
if y1 > handles.vid.Height; y1 = handles.vid.Height; end
if x1 < 1; x1 = 1; end
if x1 > handles.vid.Width; x1 = handles.vid.Width; end
handles.pnt12 = [x1, y1];
handles.edit12.String = sprintf('%.0f, %0.f',x1,y1);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[v_file,v_path] = uigetfile( ...
    {'*.avi;*.mp4;*.mov;',...
    'Video Dosyasý (*.avi;*.mp4;*.mov;)'},...
    'Deney Kaydýnýn Tutulduðu Video Dosyasýný Seçiniz.');
handles.filename = v_file(1:find(v_file=='.',1)-1);
handles.vid = VideoReader([v_path,v_file]);
handles.slider2.Max = handles.vid.Duration;
handles.slider2.Enable = 'on';
handles.slider2.Visible = 'on';
handles.axes2.Visible = 'on';
Frame = readFrame(handles.vid);
handles.Frame = Frame;
axes(handles.axes2);
imshow(Frame);
axis normal;
handles.text2.String = sprintf('t = 0 / %5.5f s',handles.vid.Duration);
handles.text2.Visible = 'on';
handles.pushbutton2.Visible = 'on';
handles.edit1.Visible = 'on';
handles.edit2.Visible = 'on';
handles.edit3.Visible = 'on';
handles.edit4.Visible = 'on';
handles.edit5.Visible = 'on';
handles.edit6.Visible = 'on';
handles.edit7.Visible = 'on';
handles.edit8.Visible = 'on';
handles.edit9.Visible = 'on';
handles.edit10.Visible = 'on';
handles.edit11.Visible = 'on';
handles.edit12.Visible = 'on';

handles.t_stop = handles.vid.Duration;
handles.t_start = 0;
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
TimeStamp = handles.slider2.Value;
handles.vid.CurrentTime = TimeStamp;
Frame = readFrame(handles.vid);
axes(handles.axes2);
imshow(Frame);
axis normal;
handles.text2.String = sprintf('t = %5.5f / %5.5f s',TimeStamp,handles.vid.Duration);
handles.Frame = Frame;
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit13.String;
handles.t_start = str2double(str);
if handles.t_start < 0; handles.t_start = 0; end
handles.slider2.Value = handles.t_start;
handles.slider2.Min = handles.t_start;
handles.vid.CurrentTime = handles.t_start;
Frame = readFrame(handles.vid);
axes(handles.axes2);
imshow(Frame);
axis normal;
handles.text2.String = sprintf('t = %5.5f / %5.5f s',handles.t_start,handles.t_stop);
handles.Frame = Frame;
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit13 as text
%        str2double(get(hObject,'String')) returns contents of edit13 as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit14_Callback(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str = handles.edit14.String;
handles.t_stop = str2double(str);
if handles.t_stop > handles.vid.Duration; handles.t_stop = handles.vid.Duration; end
handles.slider2.Value = handles.t_stop;
handles.slider2.Max = handles.t_stop;
handles.vid.CurrentTime = handles.t_stop;
Frame = readFrame(handles.vid);
axes(handles.axes2);
imshow(Frame);
axis normal;
handles.text2.String = sprintf('t = %5.5f / %5.5f s',handles.t_stop,handles.t_stop);
handles.Frame = Frame;
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edit14 as text
%        str2double(get(hObject,'String')) returns contents of edit14 as a double


% --- Executes during object creation, after setting all properties.
function edit14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
P = [handles.pnt1;handles.pnt2;handles.pnt3;handles.pnt4;handles.pnt5;...
    handles.pnt6;handles.pnt7;handles.pnt8;handles.pnt9;handles.pnt10;...
    handles.pnt11;handles.pnt12;handles.pnt1];
handles.X = P(:,1)'; handles.Y = P(:,2)';
mask = double(poly2mask(handles.X,handles.Y,handles.vid.Height,handles.vid.Width));
Frame = double(handles.Frame);
RFrame = Frame(:,:,1);
GFrame = Frame(:,:,2);
BFrame = Frame(:,:,3);
mnRFrame = RFrame.*(1-mask);
mnBFrame = BFrame.*(1-mask);
mnGFrame = GFrame.*(1-mask);
mpRFrame = RFrame.*mask;
mpBFrame = BFrame.*mask;
mpGFrame = GFrame.*mask;
mFrame = uint8(zeros(size(Frame)));
mFrame(:,:,1) = uint8(mnRFrame)/3 + uint8(mpRFrame);
mFrame(:,:,2) = uint8(mnGFrame)/3+ uint8(mpGFrame)*1.5;
mFrame(:,:,3) = uint8(mnBFrame)/3 + uint8(mpBFrame);
handles.axes3.Visible = 'on';
axes(handles.axes3);
imshow(mFrame);
hold on, plot(handles.X,handles.Y,'LineWidth',3,'Color','r'); hold off,
axis normal;
handles.pushbutton3.Visible = 'on';
handles.M = mask;

handles.text3.Visible = 'on';
handles.text4.Visible = 'on';
handles.edit13.Visible = 'on';
handles.edit14.Visible = 'on';

handles.edit13.String = sprintf('%.2f',handles.t_start);
handles.edit14.String = sprintf('%.2f',handles.t_stop);
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.edit1.Enable = 'off';
handles.edit2.Enable = 'off';
handles.edit3.Enable = 'off';
handles.edit4.Enable = 'off';
handles.edit5.Enable = 'off';
handles.edit6.Enable = 'off';
handles.edit7.Enable = 'off';
handles.edit8.Enable = 'off';
handles.edit9.Enable = 'off';
handles.edit10.Enable = 'off';
handles.edit11.Enable = 'off';
handles.edit12.Enable = 'off';
handles.edit13.Enable = 'off';
handles.edit14.Enable = 'off';
handles.slider2.Enable = 'off';
handles.pushbutton1.Enable = 'off';
handles.pushbutton2.Enable = 'off';
handles.pushbutton3.Enable = 'off';

handles.axes4.Visible = 'on';
axes(handles.axes4);
set(gca,'nextplot','replacechildren'); 

handles.vid.CurrentTime = handles.t_start;   % point to the chosen frame
vL = round((handles.t_stop-handles.t_start)*handles.vid.FrameRate);

amx = zeros(1,vL);
amn = zeros(1,vL);
bmx = zeros(1,vL);
bmn = zeros(1,vL);
t = zeros(1,vL);
xc = zeros(1,vL);
yc = zeros(1,vL);
North = zeros(1,vL);
East = zeros(1,vL);
Center = zeros(1,vL);
South = zeros(1,vL);
West = zeros(1,vL);

handles.v = VideoWriter([handles.filename,'_res.avi']);
open(handles.v);

ii = 1;
while hasFrame(handles.vid)
    frameRGB = readFrame(handles.vid);
    frameHSV = rgb2hsv(frameRGB);
    frameH = frameHSV(:,:,1);
    frameHm = double(frameH).*handles.M;
    Rat = zeros(size(frameHm));
%     Rat((frameHm>0)&(frameHm<0.2)) = 1;
    Rat((frameHm>0)&(frameHm<0.13)) = 1;
    Rat2 = imfill(Rat,8,'holes');
    se = strel('disk',3);
    marker = imdilate(Rat,se);
    [Lmarker,N] = bwlabel(marker);
    S = zeros(1,N);
    for n = 1 : N
        S(n) = sum(sum(Lmarker == n));
    end
    L = find(S==max(S),1);
    LRat = Lmarker==L;
    [a,b] = find(LRat==1);
    amx(ii) = max(a);
    amn(ii) = min(a);
    bmx(ii) = max(b);
    bmn(ii) = min(b);
    RatBox = zeros(size(LRat));
    RatBox(amn(ii),bmn(ii):bmx(ii)) = 1;
    RatBox(amx(ii),bmn(ii):bmx(ii)) = 1;
    RatBox(amn(ii):amx(ii),bmn(ii)) = 1;
    RatBox(amn(ii):amx(ii),bmx(ii)) = 1;
    img = LRat;
    [x, y] = meshgrid(1:size(img, 2), 1:size(img, 1));
    weightedx = x .* img;
    weightedy = y .* img;
    xc(ii) = sum(weightedx(:)) / sum(img(:));
    yc(ii) = sum(weightedy(:)) / sum(img(:));
    
    ShwR = frameRGB(:,:,1);
    ShwG = frameRGB(:,:,2);
    ShwB = frameRGB(:,:,3);
    ShwR(RatBox==1) = uint8(255);
    ShwG(RatBox==1) = uint8(0);
    ShwB(RatBox==1) = uint8(0);
    ShwR(fix(yc(ii))-3:fix(yc(ii))+3,...
        fix(xc(ii))-3:fix(xc(ii))+3) = uint8(0);
    ShwG(fix(yc(ii))-3:fix(yc(ii))+3,...
        fix(xc(ii))-3:fix(xc(ii))+3) = uint8(255);
    ShwB(fix(yc(ii))-3:fix(yc(ii))+3,...
        fix(xc(ii))-3:fix(xc(ii))+3) = uint8(0);
    Shw(:,:,1) = ShwR;
    Shw(:,:,2) = ShwG;
    Shw(:,:,3) = ShwB;
    
    imshow(Shw,[]);drawnow;
    if amn(ii)<max([handles.pnt12(2),handles.pnt3(2)])-10
        North(ii) = 1;
        if amx(ii)>max([handles.pnt12(2),handles.pnt3(2)])+10
            Center(ii) = 1;
        end
    end
    if bmx(ii)>min([handles.pnt6(1),handles.pnt3(1)])+10
        East(ii) = 1;
        if bmn(ii)<min([handles.pnt6(1),handles.pnt3(1)])-10
            Center(ii) = 1;
        end
    end
    if amx(ii)>max([handles.pnt9(2),handles.pnt6(2)])+10
        South(ii) = 1;
        if amn(ii)<max([handles.pnt9(2),handles.pnt6(2)])-10
            Center(ii) = 1;
        end
    end
    if bmn(ii)<min([handles.pnt12(1),handles.pnt9(1)])-10
        West(ii) = 1;
        if bmx(ii)>min([handles.pnt12(1),handles.pnt9(1)])+10
            Center(ii) = 1;
        end
    end
    t(ii) = handles.vid.CurrentTime;
    title(sprintf('Time %.2f, North %.0f, East %.0f, South %.0f, West %.0f, Center %.0f',t(ii), North(ii), East(ii), South(ii), West(ii), Center(ii)));

    handles.text2.String = sprintf('t = %5.5f / %5.5f s',t(ii),handles.t_stop);

    writeVideo(handles.v,Shw);
    ii = ii + 1;
    if ii > vL
        break;
    end
end

close(handles.v);

handles.axes5.Visible = 'on';
axes(handles.axes5);
plot(t,sgolayfilt(xc,1,51),'b',t,sgolayfilt(yc,1,51),'r')

save([handles.filename,'_res.mat'],'Center','North','West','East','South',...
    'amn','amx','bmn','bmx','xc','yc','t')
% Choose default command line output for TMaze_v1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
