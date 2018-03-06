clear;clc;
I=imread('lena.bmp','bmp');         %读取图像信息
figure;imshow(I);title('原始图片');
figure;imhist(I);title('原始图片直方图');
axis([0 255 0 4000]);
[M,N]=size(I);                      %将图像的行列赋值给M,N
t=4;    %分块大小

%% 原始图片信息熵
T1=imhist(I);   %统计图像灰度值从0~255的分布情况，存至T1
S1=sum(T1);     %计算整幅图像的灰度值
xxs1=0;
for i=1:256
    pp1=T1(i)/S1;   %每个灰度值占比，即每个灰度值的概率
    if pp1~=0
        xxs1=xxs1-pp1*log2(pp1);
    end
end

%% 原始图像相邻像素相关性分析
%{
先随机在0~M-1行和0~N-1列选中1000个像素点，
计算水平相关性时，选择每个点的相邻的右边的点；
计算垂直相关性时，选择每个点的相邻的下方的点；
计算对角线相关性时，选择每个点的相邻的右下方的点。
%}
NN=1000;    %随机取1000对像素点
x1=ceil(rand(1,NN)*(M-1));      %生成1000个1~M-1的随机整数作为行
y1=ceil(rand(1,NN)*(N-1));      %生成1000个1~N-1的随机整数作为列
EX1=0;EY1_SP=0;DX1=0;DY1_SP=0;COVXY1_SP=0;    %计算水平相关性时需要的变量
EY1_CZ=0;DY1_CZ=0;COVXY1_CZ=0;                %垂直
EY1_DJX=0;DY1_DJX=0;COVXY1_DJX=0;             %对角线
I=double(I);
for i=1:NN
    %第一个像素点的E，水平、垂直、对角线时计算得出的第一个像素点的E相同，统一用EX1表示
    EX1=EX1+I(x1(i),y1(i)); 
    %第二个像素点的E，水平、垂直、对角线的E分别对应EY1_SP、EY1_CZ、EY1_DJX
    EY1_SP=EY1_SP+I(x1(i),y1(i)+1);
    EY1_CZ=EY1_CZ+I(x1(i)+1,y1(i));
    EY1_DJX=EY1_DJX+I(x1(i)+1,y1(i)+1);
end
%统一在循环外除以像素点对数1000，可减少运算次数
EX1=EX1/NN;
EY1_SP=EY1_SP/NN;
EY1_CZ=EY1_CZ/NN;
EY1_DJX=EY1_DJX/NN;
for i=1:NN
    %第一个像素点的D，水平、垂直、对角线时计算得出第一个像素点的D相同，统一用DX表示
    DX1=DX1+(I(x1(i),y1(i))-EX1)^2;
    %第二个像素点的D，水平、垂直、对角线的D分别对应DY1_SP、DY1_CZ、DY1_DJX
    DY1_SP=DY1_SP+(I(x1(i),y1(i)+1)-EY1_SP)^2;
    DY1_CZ=DY1_CZ+(I(x1(i)+1,y1(i))-EY1_CZ)^2;
    DY1_DJX=DY1_DJX+(I(x1(i)+1,y1(i)+1)-EY1_DJX)^2;
    %两个相邻像素点相关函数的计算，水平、垂直、对角线
    COVXY1_SP=COVXY1_SP+(I(x1(i),y1(i))-EX1)*(I(x1(i),y1(i)+1)-EY1_SP);
    COVXY1_CZ=COVXY1_CZ+(I(x1(i),y1(i))-EX1)*(I(x1(i)+1,y1(i))-EY1_CZ);
    COVXY1_DJX=COVXY1_DJX+(I(x1(i),y1(i))-EX1)*(I(x1(i)+1,y1(i)+1)-EY1_DJX);
end
%统一在循环外除以像素点对数1000，可减少运算次数
DX1=DX1/NN;
DY1_SP=DY1_SP/NN;
DY1_CZ=DY1_CZ/NN;
DY1_DJX=DY1_DJX/NN;
COVXY1_SP=COVXY1_SP/NN;
COVXY1_CZ=COVXY1_CZ/NN;
COVXY1_DJX=COVXY1_DJX/NN;
%水平、垂直、对角线的相关性
RXY1_SP=COVXY1_SP/sqrt(DX1*DY1_SP);
RXY1_CZ=COVXY1_CZ/sqrt(DX1*DY1_CZ);
RXY1_DJX=COVXY1_DJX/sqrt(DX1*DY1_DJX);

%% 1.补零
%将图像的行列数都补成可以被t整除的数，t为分块的大小。
M1=mod(M,t);
N1=mod(N,t);
if M1~=0
    I(M+1:M+t-M1,:)=0;
end
if N1~=0
    I(:,N+1:N+t-N1)=0;
end
[M,N]=size(I);  %补零后的行数和列数
SUM=M*N;

%% 2.产生Logistic混沌序列
u=3.99;     %Logistic参数μ，自定为3.99
x0=sum(I(:))/(255*SUM);     %计算得出Logistic初值x0
x0=floor(x0*10^4)/10^4;     %保留4位小数
p=zeros(1,SUM+1000);        %预分配内存
p(1)=x0;
for i=1:SUM+999                 %进行SUM+999次循环，共得到SUM+1000点（包括初值）
    p(i+1)=u*p(i)*(1-p(i));
end
p=p(1001:length(p));            %去除前1000点，获得更好的随机性

%% 3.将p序列变换到0~255范围内整数，转换成M*N的二维矩阵R
p=mod(ceil(p*10^3),256);
R=reshape(p,N,M)';  %转成M行N列的随机矩阵R

%% 4.求解Chen氏超混沌系统
%求四个初值X0,Y0,Z0,H0
r=(M/t)*(N/t);      %r为分块个数
%求出四个初值
X0=sum(sum(bitand(I,3)))/(3*SUM);
Y0=sum(sum(bitand(I,12)/4))/(3*SUM);
Z0=sum(sum(bitand(I,48)/16))/(3*SUM);
H0=sum(sum(bitand(I,192)/64))/(3*SUM);
%保留四位小数
X0=floor(X0*10^4)/10^4;
Y0=floor(Y0*10^4)/10^4;
Z0=floor(Z0*10^4)/10^4;
H0=floor(H0*10^4)/10^4;
%根据初值，求解Chen氏超混沌系统，得到四个混沌序列
A=chen_output(X0,Y0,Z0,H0,r);   
X=A(:,1);
X=X(1502:length(X));        %去除前1501项，获得更好的随机性（求解陈氏系统的子函数多计算了1500点）
Y=A(:,2);
Y=Y(1502:length(Y));
Z=A(:,3);
Z=Z(1502:length(Z));
H=A(:,4);
H=H(1502:length(H));

%% 5.DNA编码
%X,Y分别决定I和R的DNA编码方式，有8种，1~8
%Z决定运算方式，有3种，0~2，0表示加，1表示减，2表示异或
%H表示DNA解码方式，有8种，1~8
X=mod(floor(X*10^4),8)+1;
Y=mod(floor(Y*10^4),8)+1;
Z=mod(floor(Z*10^4),3);
H=mod(floor(H*10^4),8)+1;
e=N/t;  %e表示每一行可以分为多少块
Q1=DNA_bian(fenkuai(t,I,1),X(1));
Q2=DNA_bian(fenkuai(t,R,1),Y(1));
Q_last=DNA_yunsuan(Q1,Q2,Z(1));
Q(1:t,1:t)=DNA_jie(Q_last,H(1));
for i=2:r
    Q1=DNA_bian(fenkuai(t,I,i),X(i));   %对原始图像每一个分块按X对应的序号进行DNA编码
    Q2=DNA_bian(fenkuai(t,R,i),Y(i));   %对R的每一个分块按Y对应的序号进行DNA编码
    Q3=DNA_yunsuan(Q1,Q2,Z(i));         %对上面两个编码好的块按Z对应的序号进行DNA运算
    Q4=DNA_yunsuan(Q3,Q_last,Z(i));     %运算结果在和前一块按Z对应的序号再一次进行运算，称为扩散
    Q_last=Q4;
    xx=floor(i/e)+1;
    yy=mod(i,e);
    if yy==0
        xx=xx-1;
        yy=e;
    end
    Q((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_jie(Q4,H(i));    %将每一块合并成完整的图Q
end
Q=uint8(Q);
imwrite(Q,'加密后的lena.bmp','bmp');        
figure;imshow(Q);title('加密后图片');
figure;imhist(Q);title('加密后直方图');
axis([0 255 0 2000]);

%% 加密后信息熵
T2=imhist(Q);
S2=sum(T2);
xxs2=0;
for i=1:256
    pp2=T2(i)/S2;
    if pp2~=0
        xxs2=xxs2-pp2*log2(pp2);
    end
end

%% 加密图像相邻图像相关性分析
%{
先随机在0~M-1行和0~N-1列选中1000个像素点，
计算水平相关性时，选择每个点的相邻的右边的点；
计算垂直相关性时，选择每个点的相邻的下方的点；
计算对角线相关性时，选择每个点的相邻的右下方的点。
%}
Q=double(Q);
EX2=0;EY2_SP=0;DX2=0;DY2_SP=0;COVXY2_SP=0;    %水平
EY2_CZ=0;DY2_CZ=0;COVXY2_CZ=0;    %垂直
EY2_DJX=0;DY2_DJX=0;COVXY2_DJX=0;   %对角线
for i=1:NN
    %第一个像素点的E，水平、垂直、对角线时计算得出的第一个像素点的E相同，统一用EX2表示
    EX2=EX2+Q(x1(i),y1(i));
    %第二个像素点的E，水平、垂直、对角线的E分别对应EY2_SP、EY2_CZ、EY2_DJX
    EY2_SP=EY2_SP+Q(x1(i),y1(i)+1);
    EY2_CZ=EY2_CZ+Q(x1(i)+1,y1(i));
    EY2_DJX=EY2_DJX+Q(x1(i)+1,y1(i)+1);
end
%统一在循环外除以像素点对数1000，可减少运算次数
EX2=EX2/NN;
EY2_SP=EY2_SP/NN;
EY2_CZ=EY2_CZ/NN;
EY2_DJX=EY2_DJX/NN;
for i=1:NN
    %第一个像素点的D，水平、垂直、对角线时计算得出第一个像素点的D相同，统一用DX2表示
    DX2=DX2+(Q(x1(i),y1(i))-EX2)^2;
    %第二个像素点的D，水平、垂直、对角线的D分别对应DY2_SP、DY2_CZ、DY2_DJX
    DY2_SP=DY2_SP+(Q(x1(i),y1(i)+1)-EY2_SP)^2;
    DY2_CZ=DY2_CZ+(Q(x1(i)+1,y1(i))-EY2_CZ)^2;
    DY2_DJX=DY2_DJX+(Q(x1(i)+1,y1(i)+1)-EY2_DJX)^2;
    %两个相邻像素点相关函数的计算，水平、垂直、对角线
    COVXY2_SP=COVXY2_SP+(Q(x1(i),y1(i))-EX2)*(Q(x1(i),y1(i)+1)-EY2_SP);
    COVXY2_CZ=COVXY2_CZ+(Q(x1(i),y1(i))-EX2)*(Q(x1(i)+1,y1(i))-EY2_CZ);
    COVXY2_DJX=COVXY2_DJX+(Q(x1(i),y1(i))-EX2)*(Q(x1(i)+1,y1(i)+1)-EY2_DJX);
end
%统一在循环外除以像素点对数1000，可减少运算次数
DX2=DX2/NN;
DY2_SP=DY2_SP/NN;
DY2_CZ=DY2_CZ/NN;
DY2_DJX=DY2_DJX/NN;
COVXY2_SP=COVXY2_SP/NN;
COVXY2_CZ=COVXY2_CZ/NN;
COVXY2_DJX=COVXY2_DJX/NN;
%水平、垂直、对角线的相关性
RXY2_SP=COVXY2_SP/sqrt(DX2*DY2_SP);
RXY2_CZ=COVXY2_CZ/sqrt(DX2*DY2_CZ);
RXY2_DJX=COVXY2_DJX/sqrt(DX2*DY2_DJX);

%% 输出数据信息
disp('加密成功');  
disp(['密钥1：μ=',num2str(u),'     密钥2：x0=',num2str(x0),'    密钥3：x(0)=',num2str(X0)]);
disp(['密钥4：y(0)=',num2str(Y0),'  密钥2：z(0)=',num2str(Z0),'   密钥3：h(0)=',num2str(H0)]);
disp(['原始图片信息熵=',num2str(xxs1),'  加密后图片信息熵=',num2str(xxs2)]);
disp(['原始图片相关性：','  水平相关性=',num2str(RXY1_SP),'  垂直相关性=',num2str(RXY1_CZ),'  对角线相关性=',num2str(RXY1_DJX)]);
disp(['加密图片相关性：','  水平相关性=',num2str(RXY2_SP),'  垂直相关性=',num2str(RXY2_CZ),'  对角线相关性=',num2str(RXY2_DJX)]);