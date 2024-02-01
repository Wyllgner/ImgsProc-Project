package com.example.customizaesdeimagens.activity.activity;

import static com.example.customizaesdeimagens.activity.model.Funcoes.getFuncoesParametro;
import static com.example.customizaesdeimagens.activity.model.Funcoes.processarImagemSelecionada;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.customizaesdeimagens.R;
import com.example.customizaesdeimagens.activity.model.Funcoes;

import java.io.IOException;

public class MainActivity2 extends AppCompatActivity {

    private ImageView imageView;
    private static final int PICK_IMAGE_REQUEST = 1;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        imageView = findViewById(R.id.imageView);
        Button btnSelectImage = findViewById(R.id.btnSelectImage);
        Button btnProcessImage = findViewById(R.id.btnProcessImage);

        // Botão para selecionar a imagem da galeria
        btnSelectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                abrirGaleria();
            }
        });

        // Botão para processar a imagem
        btnProcessImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Obtenha o bitmap da imagem exibida no ImageView
                Bitmap imagemBitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
                processarImagemSelecionada(imagemBitmap);
            }
        });
    }

    // Método para abrir a galeria
    private void abrirGaleria() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
    }

    // Método chamado quando a imagem é selecionada da galeria
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            // Obter o URI da imagem selecionada
            try {
                Bitmap imagemBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
                // Exibir a imagem selecionada
                imageView.setImageBitmap(imagemBitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
