package com.example.customizaesdeimagens.activity.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.customizaesdeimagens.R;
import com.example.customizaesdeimagens.activity.activity.RecyclerViewInterface;
import com.example.customizaesdeimagens.activity.model.Funcoes;

import java.util.List;

public class Adapter extends RecyclerView.Adapter<Adapter.MViewHolder> {
    private final RecyclerViewInterface recyclerViewInterface;
    private List<Funcoes> listaFuncoes;
    public Adapter(List<Funcoes> lista, RecyclerViewInterface recyclerViewInterface) {
        this.listaFuncoes = lista;
        this.recyclerViewInterface = recyclerViewInterface;
    }

    @NonNull
    @Override
    public MViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) { // cria listas
        View itLista = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.lista, parent, false);


        return new MViewHolder(itLista, recyclerViewInterface);
    }

    @Override
    public void onBindViewHolder(@NonNull MViewHolder holder, int position) { // mostra as lista
        Funcoes funcao = listaFuncoes.get(position);
        holder.titulo.setText(funcao.getTituloFuncao());

    }

    @Override
    public int getItemCount() { //qtd de itens exibidos
        return listaFuncoes.size();
    }

    public class MViewHolder extends RecyclerView.ViewHolder{
        TextView titulo;
        public MViewHolder(@NonNull View itemView, RecyclerViewInterface recyclerViewInterface) {
            super(itemView);
            titulo = itemView.findViewById(R.id.Titulo);
            itemView.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    if(Adapter.this.recyclerViewInterface != null){
                        int pos = getAdapterPosition();

                        if(pos != RecyclerView.NO_POSITION){
                            Adapter.this.recyclerViewInterface.onItemClick(pos);
                        }
                    }
                }
            });
        }
    }

}
