package com.example.customizaesdeimagens.activity.activity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import com.example.customizaesdeimagens.R;

public class AcitivityInicio extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_inicio);

        // Localize os botões pelos IDs
        Button testarButton = findViewById(R.id.Testar);

        // Defina os ouvintes de clique para os botões
        testarButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Ao clicar no botão "Testar", inicie a MainActivity1
                Intent intent = new Intent(AcitivityInicio.this, MainActivity.class);
                startActivity(intent);
            }
        });

    }
}
