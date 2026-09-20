% --- Configuration ---
num_simulations = 100;
output_folder = 'dataset_panto';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

etats = {'Sain', 'Fissure', 'Fuite'};

for e = 1:length(etats)
    etat_actuel = etats{e};

    for i = 1:num_simulations
        % --- Logique des paramètres ---
        if strcmp(etat_actuel, 'Sain')
            num1 = 141.6180; den1 = [5.0139 41.3247 69.0255];
            num2 = 245.1882; den2 = [15.0293 123.8716 206.9052];
            num3 = 252.4573; den3 = [15.9337 131.3254 219.3554];
            num4 = 193.6660; den4 = [9.3766 77.2822 129.0860];
            num5 = 150.5337; den5 = [5.6650 46.6917 77.9901];

        elseif strcmp(etat_actuel, 'Fissure')
            num1 = 141.6180 * 0.8; den1 = [5.0139 41.3247 69.0255];
            num2 = 245.1882; den2 = [15.0293 123.8716 206.9052];
            num3 = 252.4573; den3 = [15.9337 131.3254 219.3554];
            num4 = 193.6660; den4 = [9.3766 77.2822 129.0860];
            num5 = 150.5337; den5 = [5.6650 46.6917 77.9901];

        else % Fuite
            num1 = 141.6180; den1 = [5.0139 41.3247 69.0255] * 1.2;
            num2 = 245.1882; den2 = [15.0293 123.8716 206.9052];
            num3 = 252.4573; den3 = [15.9337 131.3254 219.3554];
            num4 = 193.6660; den4 = [9.3766 77.2822 129.0860];
            num5 = 150.5337; den5 = [5.6650 46.6917 77.9901];
        end

        % Export vers le Base Workspace
        assignin('base', 'num1', num1); assignin('base', 'den1', den1);
        assignin('base', 'num2', num2); assignin('base', 'den2', den2);
        assignin('base', 'num3', num3); assignin('base', 'den3', den3);
        assignin('base', 'num4', num4); assignin('base', 'den4', den4);
        assignin('base', 'num5', num5); assignin('base', 'den5', den5);

        % --- Lancer la simulation ---
        out = sim('modelisation_simu');

        % --- Extraction sécurisée des données ---
        if isprop(out, 'data_simulation')
            data_to_save = out.data_simulation;
        else
            error('La variable "data_simulation" n''est pas trouvée dans la sortie de simulation.');
        end

        % --- Sauvegarde ---
        filename = sprintf('%s/%s_%d.csv', output_folder, etat_actuel, i);
        writematrix(data_to_save, filename); % 'writematrix' est recommandé à la place de 'csvwrite'

        fprintf('Simulation %d/%d pour %s terminée.\n', i, num_simulations, etat_actuel);
    end
end