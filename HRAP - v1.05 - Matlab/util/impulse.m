%-----------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  impulse
% 
% Purpose:  identify impulse class of motor based on total delvered impulse
%
%-----------------------------------------------------------------------------

function [motorClass,percent] = impulse(totalImpulse)

    if totalImpulse <= 1.25
        motorClass = 'A';
        percent = 100.*(totalImpulse - 0)./(1.25);
    elseif totalImpulse > 1.25 && totalImpulse <= 5
        motorClass = 'B';
        percent = 100.*(totalImpulse - 1.25)./(2.5);
    elseif totalImpulse > 5 && totalImpulse <= 10
        motorClass = 'C';
        percent = 100.*(totalImpulse - 5)./(5);
    elseif totalImpulse > 10 && totalImpulse <= 20
        motorClass = 'D';
        percent = 100.*(totalImpulse - 10)./(10);
    elseif totalImpulse > 20 && totalImpulse <= 40
        motorClass = 'E';
        percent = 100.*(totalImpulse - 20)./(20);
    elseif totalImpulse > 40 && totalImpulse <= 80
        motorClass = 'F';
        percent = 100.*(totalImpulse - 40)./(40);
    elseif totalImpulse > 80 && totalImpulse <= 160
        motorClass = 'G';
        percent = 100.*(totalImpulse - 80)./(80);
    elseif totalImpulse > 160 && totalImpulse <= 320
        motorClass = 'H';
        percent = 100.*(totalImpulse - 160)./(160);
    elseif totalImpulse > 320 && totalImpulse <= 640
        motorClass = 'I';
        percent = 100.*(totalImpulse - 320)./(320);
    elseif totalImpulse > 640 && totalImpulse <= 1280
        motorClass = 'J';
        percent = 100.*(totalImpulse - 640)./(640);
    elseif totalImpulse > 1280 && totalImpulse <= 2560
        motorClass = 'K';
        percent = 100.*(totalImpulse - 1280)./(1280);
    elseif totalImpulse > 2560 && totalImpulse <= 5120
        motorClass = 'L';
        percent = 100.*(totalImpulse - 2560)./(2560);
    elseif totalImpulse > 5120 && totalImpulse <= 10240
        motorClass = 'M';
        percent = 100.*(totalImpulse - 5120)./(5120);
    elseif totalImpulse > 10240 && totalImpulse <= 20480
        motorClass = 'N';
        percent = 100.*(totalImpulse - 10240)./(10240);
    elseif totalImpulse > 20480 && totalImpulse <= 40960
        motorClass = 'O';
        percent = 100.*(totalImpulse - 20480)./(20480);
    elseif totalImpulse > 40960 && totalImpulse <= 81920
        motorClass = 'P';
        percent = 100.*(totalImpulse - 40960)./(40960);
    elseif totalImpulse > 81920 && totalImpulse <= 163840
        motorClass = 'Q';
        percent = 100.*(totalImpulse - 81920)./(81920);
    elseif totalImpulse > 163840 && totalImpulse <= 327680
        motorClass = 'R';
        percent = 100.*(totalImpulse - 163840)./(163840);
    elseif totalImpulse > 327680 && totalImpulse <= 655369
        motorClass = 'S';
        percent = 100.*(totalImpulse - 327680)./(327680);
    elseif totalImpulse > 655360 && totalImpulse <= 1310720
        motorClass = 'T';
        percent = 100.*(totalImpulse - 655360)./(655360);
    else
        motorClass = NaN;
        percent = NaN;
    end

end

%-----------------------------------------------------------------------------
% END OF PROGRAM
%-----------------------------------------------------------------------------