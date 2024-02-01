package com.example.customizaesdeimagens.activity.model;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class Funcoes {
    private String tituloFuncao;
    private static int position;
    public static final int PICK_IMAGE_REQUEST = 1; // Código de solicitação para a seleção de imagem

    public Funcoes() {
    }

    public Funcoes(String tituloFuncao, int position) {
        this.tituloFuncao = tituloFuncao;
        Funcoes.position = position;
    }

    public String getTituloFuncao() {
        return tituloFuncao;
    }

    public void setTituloFuncao(String tituloFuncao) {
        this.tituloFuncao = tituloFuncao;
    }

    public static int getPosition() {
        return position;
    }

    public void setPosition(int position) {
        Funcoes.position = position;
    }

    public static void getFuncoesParametro(Activity activity) {
        //if (getPosition() == 0) {
        // Criar uma Intent para a seleção de imagem da galeria
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");

        // Iniciar a activity de seleção de imagem
        activity.startActivityForResult(intent, PICK_IMAGE_REQUEST);
        // }
    }

    public static void processarImagemSelecionada(Bitmap imagemBitmap) {
        // Converter a imagem para bytes
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        imagemBitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();
        String encodedImage = Base64.encodeToString(byteArray, Base64.DEFAULT);

        // Enviar a imagem para o servidor Python
        new EnviarImagemParaServidor().execute(encodedImage);
    }

    // AsyncTask para enviar a imagem para o servidor Python
    private static class EnviarImagemParaServidor extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... params) {
            try {
                String imagemBase64 = params[0];

                // Configurar a URL do servidor Python
                URL url = new URL("http://10.0.2.2:5000/process_image"); // Substitua pelo IP do seu servidor
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setRequestProperty("Content-Type", "multipart/form-data;boundary=*****");
                connection.setDoOutput(true);

                // Construir o corpo da solicitação
                DataOutputStream outputStream = new DataOutputStream(connection.getOutputStream());
                String boundary = "*****";
                String lineEnd = "\r\n";

                outputStream.writeBytes("--" + boundary + lineEnd);
                outputStream.writeBytes("Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"" + lineEnd);
                outputStream.writeBytes("Content-Type: image/jpeg" + lineEnd);
                outputStream.writeBytes(lineEnd);
                outputStream.write(Base64.decode(imagemBase64, Base64.DEFAULT));
                outputStream.writeBytes(lineEnd);
                outputStream.writeBytes("--" + boundary + "--" + lineEnd);

                outputStream.flush();
                outputStream.close();

                // Obter a resposta do servidor
                int responseCode = connection.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    // Ler a resposta do servidor
                    InputStream in = connection.getInputStream();
                    // Processar a resposta conforme necessário
                    // ...

                    // Retornar uma mensagem de sucesso ou a resposta do servidor (dependendo do seu caso)
                    return "Sucesso";
                } else {
                    return "Erro: " + responseCode;
                }
            } catch (Exception e) {
                e.printStackTrace();
                return "Erro: " + e.getMessage();
            }
        }

        @Override
        protected void onPostExecute(String result) {
            // Manipular o resultado conforme necessário
            Log.d("Resultado do servidor", result);
        }
    }
}
