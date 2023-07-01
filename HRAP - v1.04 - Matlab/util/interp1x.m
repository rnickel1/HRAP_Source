%-----------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program: interp1x
% 
% Purpose: high efficiency 1-dimensional linear interpolation for point xi given 1xn arrays X and Y
%          up to 10x faster than MATLAB/Octave's built in interp1 command
%
% Input:   row vectors X and Y of length n, point xi
%
% Output:  linearly interpolated value of yi at point xi
%
% Variables:
%
% X:       row vector of length n
% Y:       row vector of length n
% xi:      desired x value for interpolation
% yi:      desired y value for interpolation
% zi:      interpolated Z value at point (xi,yi)
% k:       indice of X immediately preceeding desired point xi
% yi:      interpolated value of Y at point xi
%
%-----------------------------------------------------------------------------

function [yi] = interp1x(X,Y,xi)

  if xi <= X(1)

    yi = Y(1); % If xi is lower than lowest X value, it defaults to the closest value of X

  elseif xi >= X(length(X))

    yi = Y(length(Y)); % If xi is greater than the highest X value, it defaults to the closest value of X

  else

    [k,k] = histc(xi,X); % finds index of array X immediately preceeding xi

    yi = ((Y(k+1)-Y(k))./(X(k+1)-X(k))).*(xi - X(k)) + Y(k); % interpolated value of y at point xi

  end
 
end