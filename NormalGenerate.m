clear all
close all

tic

%Depth_Img = double(imread('RGB_Berkley/img_0098_abs_smooth.png'));
Original_Img = imread('12.jpg');
%Depth_Img(Depth_Img == 0) = 0;
%height = 480;
%width = 640;
%CONSTANT = 570.3;
%Units = 1000;

%Convert to Point Cloud
%PCloud = zeros(height,width,3);
%PCloud(:,:,1) = ones(height,1)*(1:width) + (1-1) - 320;
%PCloud(:,:,1) = PCloud(:,:,1).*Depth_Img/CONSTANT/Units;
%PCloud(:,:,2) = (1:height)'*ones(1,width) + (1-1) - 240;
%PCloud(:,:,2) = PCloud(:,:,2).*Depth_Img/CONSTANT/Units;
%PCloud(:,:,3) = Depth_Img/CONSTANT;

i = 480;
j = 640;
k = 3;
VectorCloud = (reshape(Original_Img,i*j,k))';
NeighbourSize = 150;
CloudSize = i*j;
%Compute Normal Map
Normal_Map = zeros(3,CloudSize);

SearchNeighbors = transpose(knnsearch(transpose(VectorCloud),transpose(VectorCloud),'k',NeighbourSize+1));
for l=1:CloudSize
    Vector = VectorCloud(:,SearchNeighbors(2:end,l));
    [V,Diag] = eig(2*cov(double((Vector'))));
    [~, index] = min(diag(Diag));
    Normal_Map(:,l) = V(:,index);
end

Normal_Map = Normal_Map';
Normals(:,:,1) = reshape(Normal_Map(:,1),i,j);
Normals(:,:,2) = reshape(Normal_Map(:,2),i,j);
Normals(:,:,3) = reshape(Normal_Map(:,3),i,j);

%Smooth Normal Map by Using a Bilateral Filter

s = 7;
spatialfilt = s/2;
color = 2.5;
Gaussian = fspecial('gaussian',s,color);
rad = floor(s/2);
Smooth_Map = zeros(i,j,3);

for m=1:i
    a = max(m-rad,1);
    b = min(m+rad,i);
    outboxt = (max(m-rad,1)) - (m - rad);
    outboxb = (m + rad) - (min(m+rad,i));
    for n = 1:j
        outboxl = max(n - rad,1) - (n - rad);
        outboxr = (n + rad) - min(n+rad,j);
        c = max(n - rad,1);
        d = min(n + rad,j);
        window = Normals(a:b,c:d,[1 2 3]);
        middlepix = Normals(m,n,[1 2 3]);
        FiltExp = Gaussian((1+outboxt):(s-outboxb),(1+outboxl):(s - outboxr));
        Red = window(:,:,1) - middlepix(:,:,1);
        Green = window(:,:,2) - middlepix(:,:,2);
        Blue = window(:,:,3) - middlepix(:,:,3);
        SumSquare = ((Red.*Red) + (Green.*Green) + (Blue.*Blue))/3;
        Intensity = exp(-SumSquare/(color*color));
        Filter = Intensity.*FiltExp;
        Filter_Norm = Filter/sum(Filter(:));
        TRed = window(:,:,1).*Filter_Norm;
        TGreen = window(:,:,2).*Filter_Norm;
        TBlue = window(:,:,3).*Filter_Norm;
        Smooth_Map(m,n,1) = sum(TRed(:));
        Smooth_Map(m,n,2) = sum(TGreen(:));
        Smooth_Map(m,n,3) = sum(TBlue(:));
    end
end

subplot(121),imshow(Normals);
subplot(122),imshow(Smooth_Map);

imwrite(Normals,'C:/Users/Pavan/Documents/Visual Studio 2012/Projects/open/Debug/img_0098_noisy_normal_map.png','png');
imwrite(Smooth_Map,'C:/Users/Pavan/Documents/Visual Studio 2012/Projects/open/Debug/img_0098_smooth_normal_map.png','png');

toc
