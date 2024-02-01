package com.example.customizaesdeimagens.activity.activity;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.AdapterView;
import android.widget.LinearLayout;

import com.example.customizaesdeimagens.R;
import com.example.customizaesdeimagens.activity.RecyclerItemClickListener;
import com.example.customizaesdeimagens.activity.adapter.Adapter;
import com.example.customizaesdeimagens.activity.model.Funcoes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements RecyclerViewInterface {

    private RecyclerView recyclerView;
    private List<Funcoes> listaFuncoes = new ArrayList<>();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        recyclerView = findViewById(R.id.recyclerView);

        //listagem de itens
        this.criarFuncoes();

        // configurar adapter
        Adapter adapter = new Adapter(listaFuncoes, this);



        // configurar RecyclerView
        RecyclerView.LayoutManager layoutManager = new LinearLayoutManager(getApplicationContext());
        recyclerView.setLayoutManager(layoutManager);
        recyclerView.setHasFixedSize(true);
        recyclerView.addItemDecoration(new DividerItemDecoration(this, LinearLayout.VERTICAL));
        recyclerView.setAdapter(adapter);

        // evento de clique
        recyclerView.addOnItemTouchListener(
            new RecyclerItemClickListener(
                    getApplicationContext(),
                    recyclerView,
                    new RecyclerItemClickListener.OnItemClickListener() {
                        @Override
                        public void onItemClick(View view, int position) {

                        }

                        @Override // nada acontece
                        public void onLongItemClick(View view, int position) {

                        }

                        @Override
                        public void onItemClick(AdapterView<?> parent, View view, int position, long id) {

                        }
                    }
            )
        );
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == Funcoes.PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            // Obter o URI da imagem selecionada
            Uri selectedImageUri = data.getData();

            try {
                // Converter o URI em Bitmap
                Bitmap imagemBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);

                // Criar uma instância de Funcoes e chamar o método para processar a imagem selecionada
                Funcoes funcoes = new Funcoes();
                funcoes.processarImagemSelecionada(imagemBitmap);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    public void criarFuncoes(){
        Funcoes funcao = new Funcoes("Filtro médio", 0);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("High Boost", 1);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Controle de contraste", 2);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Alongamento de contraste", 3);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Expansão de histograma", 4);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Transformação logarítmica", 5);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Filtro de mediana", 6);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Cisalhamento", 7);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Negativo", 8);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Dissolução cruzada não uniforme", 9);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Transformação de potência", 10);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Rebound", 11);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Rotação", 12);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Detecção de borda Sobel", 13);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Limiarização", 14);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Dissolução cruzada uniforme", 15);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Zoom na imagem", 16);
        this.listaFuncoes.add(funcao);

        funcao = new Funcoes("Zoom out na imagem", 17);
        this.listaFuncoes.add(funcao);
    }


    @Override
    public void onItemClick(int position) {
        Intent intent = new Intent(MainActivity.this, MainActivity2.class);
        intent.putExtra("POSITION", position);
        startActivity(intent);

    }

    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {
        super.onPointerCaptureChanged(hasCapture);
    }
}