%--------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  interp2x
% 
% Purpose:  high efficiency 2-dimensional linear interpolation for point
%           (xi,yi) given 1xn array X, mx1 array Y, and mxn array Z
%
% Input:    row vector X of length n of increasing numerical value, column
%           vector Y of length m of increasing numerical value, nxm array
%           Z, and points xi and yi at which to interpolate
%
% Output:   linearly interpolated value of Z at point xi, yi
%
% Note:     for values of xi or yi outside of range of X or Y, interpolation
%           will default to the maximum or minimum value of X or Y
%
% Limits:   X and Y must be monotonically increasing row and column vectors
%
% Variables:
%
% X:        row vector of length n
% Y:        column vector of length m
% Z:        array with dimensions nxm corresponding to X and Y
% xi:       desired x value for interpolation
% yi:       desired y value for interpolation
% zi:       interpolated Z value at point (xi,yi)
% k:        indice of X immediately preceeding desired point xi
% l:        indice of Y immediately preceeding desired point yi
% Z1:       array of Z values for X(k) and all Y values
% zi1:      interpolated value of Z between Y(l) and Y(l+1)
% Z2:       array of Z values for X(k+1) and all Y values
% zi2:      interpolated value of Z between Y(l) and Y(l+1)
% Z3:       interpolated values of Z corresponding to (X(k),yi) and
%           (X(k+1),yi)
% zi:       interpolated value of Z corresponding to (xi,yi)
%
%--------------------------------------------------------------------------

function [zi] = interp2x(X,Y,Z,xi,yi)

  % find nearest indices below xi and yi

    if xi >= X(end)

      k = length(X);

    elseif xi <= X(1)

      k = 1;
      
    else

      [~,k] = histc(xi,X); % finds index of array X immediately preceeding xi
      
    end

    if yi >= Y(end)

      l = length(Y);

    elseif yi <= Y(1)

      l = 1;

    else

      [~,l] = histc(yi,Y); % finds index of array Y immediately preceeding yi

    end

    % isolate lower row of Z for interpolation

    if xi >= X(end)

      Z1 = Z(:,length(X));

    elseif xi <= X(1)

      Z1 = Z(:,1);

    else

      Z1 = Z(:,k);

    end

    if yi >= Y(end)

      zi1 = Z1(l);

    elseif yi <= Y(1)

      zi1 = Z1(1);

    else

      zi1 = ((Z1(l+1)-Z1(l))./(Y(l+1)-Y(l))).*(yi - Y(l)) + Z1(l); % interpolated value of Z1 at yi

    end

    % isolate upper row of Z for interpolation

    if xi >= X(end)

      Z2 = Z(:,length(X));

    elseif xi < X(1)

      Z2 = Z(:,1);

    else

      Z2 = Z(:,k+1);

    end

    % Second interpolation

    if yi >= Y(end)

      zi2 = Z2(l);

    else

      zi2 = ((Z2(l+1)-Z2(l))./(Y(l+1)-Y(l))).*(yi - Y(l)) + Z2(l); % interpolated value of Z2 at yi

    end

    % Third Interpolation

    Z3 = [zi1 zi2]; % upper and lower bounds for zi

    if xi >= X(end)

      zi = Z3(2);

    elseif xi <= X(1)

      zi = Z3(1);

    else 

      zi = ((Z3(2) - Z3(1))./(X(k+1) - X(k))).*(xi - X(k)) + Z3(1); % interpolated value of Z at (xi,yi)

    end

end