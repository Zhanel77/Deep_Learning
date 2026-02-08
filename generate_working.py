"""
РАБОЧИЙ ГЕНЕРАТОР ФОТОК ДЛЯ VAE И GAN
Запуск: python generate_working.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from PIL import Image

# Добавляем текущую директорию в путь
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

class FaceGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"⚡ Устройство: {self.device}")
        
        # Загружаем модели
        self.vae = self._load_vae()
        self.gan = self._load_gan()
        
        print("\n✅ Все модели загружены успешно!")
        print(f"   Атрибуты: {self.attr_names}")
        print("   Порядок: [Улыбка, Очки, Пол, Молодой]")
        print("            [1=да/муж, 0=нет/жен, 0.5=смешанно]")
    
    def _load_vae(self):
        """Загружаем VAE модель"""
        print("\n📥 Загружаем VAE...")
        
        # Импортируем архитектуру
        from src.models.vae import CVAE
        
        # Создаем модель
        vae = CVAE(
            img_channels=3,
            img_size=64,
            attr_dim=4,
            latent_dim=128,
            base_ch=64
        )
        
        # Загружаем веса
        checkpoint = torch.load(
            "runs/cvae/weights/cvae_last.pt",
            map_location=self.device
        )
        
        # Извлекаем атрибуты
        self.attr_names = checkpoint.get('attr_names', ['Smiling', 'Eyeglasses', 'Male', 'Young'])
        
        # Загружаем веса
        vae.load_state_dict(checkpoint['model_state'])
        vae.to(self.device).eval()
        
        return vae
    
    def _load_gan(self):
        """Загружаем GAN модель"""
        print("\n📥 Загружаем GAN...")
        
        # Импортируем архитектуру
        from src.models.gan import ConditionalGenerator
        
        # Создаем модель (используем ту же архитектуру, что и в обучении)
        gan = ConditionalGenerator(
            z_dim=128,
            attr_dim=4,
            base_ch=64,
            img_ch=3
        )
        
        # Загружаем веса
        checkpoint = torch.load(
            "runs/cgan_wgangp/weights/cgan_best.pt",
            map_location=self.device
        )
        
        # Загружаем веса генератора
        gan.load_state_dict(checkpoint['G'])
        gan.to(self.device).eval()
        
        return gan
    
    def generate_face(self, attrs, model='both', num_samples=1, seed=None):
        """
        Генерация лица по атрибутам
        
        Args:
            attrs: список из 4 чисел [улыбка, очки, пол, возраст]
            model: 'vae', 'gan', или 'both'
            num_samples: количество образцов
            seed: фиксирует случайность для воспроизводимости
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Подготавливаем атрибуты
        attrs_tensor = self._prepare_attributes(attrs, num_samples)
        
        results = {}
        
        if model in ['vae', 'both']:
            z = torch.randn(num_samples, 128, device=self.device)
            with torch.no_grad():
                results['vae'] = self.vae.decode(z, attrs_tensor)
        
        if model in ['gan', 'both']:
            z = torch.randn(num_samples, 128, device=self.device)
            with torch.no_grad():
                results['gan'] = self.gan(z, attrs_tensor)
        
        return results
    
    def _prepare_attributes(self, attrs, num_samples):
        """Подготовка тензора атрибутов"""
        # Проверяем входные данные
        if isinstance(attrs, list):
            if len(attrs) != 4:
                raise ValueError(f"Ожидается 4 атрибута, получено {len(attrs)}")
            attrs_tensor = torch.tensor(attrs, device=self.device).float()
        else:
            attrs_tensor = attrs.to(self.device).float()
        
        # Добавляем размерность батча если нужно
        if len(attrs_tensor.shape) == 1:
            attrs_tensor = attrs_tensor.unsqueeze(0)
        
        # Повторяем для нескольких образцов
        if attrs_tensor.shape[0] == 1 and num_samples > 1:
            attrs_tensor = attrs_tensor.repeat(num_samples, 1)
        
        return attrs_tensor
    
    def tensor_to_image(self, tensor):
        """
        Конвертация тензора в изображение PIL
        
        Args:
            tensor: (B, 3, H, W) или (3, H, W) в диапазоне [-1, 1]
        """
        # Убираем размерность батча если нужно
        if len(tensor.shape) == 4:
            tensor = tensor[0].detach().cpu()
        else:
            tensor = tensor.detach().cpu()
        
        # Конвертация в numpy
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        
        # Масштабирование [-1, 1] -> [0, 255]
        img_np = (img_np + 1) * 127.5
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_np)
    
    def save_image(self, image_tensor, path):
        """Сохранение изображения"""
        img = self.tensor_to_image(image_tensor)
        img.save(path)
        return img
    
    def show_face(self, image_tensor, title=""):
        """Показ одного лица"""
        img = self.tensor_to_image(image_tensor)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title, fontsize=12)
        plt.show()
    
    def show_comparison(self, vae_tensor, gan_tensor, attrs, description=""):
        """Показ сравнения VAE vs GAN"""
        vae_img = self.tensor_to_image(vae_tensor)
        gan_img = self.tensor_to_image(gan_tensor)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(vae_img)
        axes[0].set_title(f"VAE\n{description}", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(gan_img)
        axes[1].set_title(f"GAN\n{description}", fontsize=12)
        axes[1].axis('off')
        
        # Добавляем атрибуты под изображениями
        attr_text = f"Атрибуты: {attrs}"
        plt.figtext(0.5, 0.01, attr_text, ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

def demo_basic_generation():
    """Базовая демонстрация генерации"""
    print("=" * 70)
    print("🎭 ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЛИЦ")
    print("=" * 70)
    
    # Создаем генератор
    gen = FaceGenerator()
    
    # Создаем папку для результатов
    output_dir = "generated_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Результаты будут сохранены в: {output_dir}/")
    
    # Примеры атрибутов для генерации
    examples = [
        {
            "attrs": [1, 0, 1, 1],  # Улыбающийся молодой мужчина без очков
            "name": "smiling_young_man",
            "desc": "Улыбающийся молодой мужчина"
        },
        {
            "attrs": [1, 0, 0, 1],  # Улыбающаяся молодая женщина без очков
            "name": "smiling_young_woman", 
            "desc": "Улыбающаяся молодая женщина"
        },
        {
            "attrs": [0, 1, 1, 0],  # Серьезный пожилой мужчина в очках
            "name": "serious_old_man_glasses",
            "desc": "Серьезный пожилой мужчина в очках"
        },
        {
            "attrs": [0, 1, 0, 0],  # Серьезная пожилая женщина в очках
            "name": "serious_old_woman_glasses",
            "desc": "Серьезная пожилая женщина в очках"
        },
        {
            "attrs": [0.5, 0.5, 0.5, 0.5],  # Смешанные атрибуты
            "name": "mixed_attributes",
            "desc": "Лицо со смешанными атрибутами"
        }
    ]
    
    print("\n🎨 Генерируем примеры...")
    print("-" * 70)
    
    all_results = []
    
    for i, example in enumerate(examples):
        print(f"\n{i+1}. {example['desc']}")
        print(f"   Атрибуты: {example['attrs']}")
        
        try:
            # Генерация
            results = gen.generate_face(
                attrs=example['attrs'],
                model='both',
                num_samples=1,
                seed=42 + i  # для воспроизводимости
            )
            
            # Сохранение
            if 'vae' in results:
                vae_img = gen.save_image(
                    results['vae'],
                    f"{output_dir}/vae_{example['name']}.png"
                )
            
            if 'gan' in results:
                gan_img = gen.save_image(
                    results['gan'],
                    f"{output_dir}/gan_{example['name']}.png"
                )
            
            # Сохраняем для отображения
            if 'vae' in results and 'gan' in results:
                all_results.append({
                    'vae': results['vae'],
                    'gan': results['gan'],
                    'desc': example['desc'],
                    'attrs': example['attrs']
                })
            
            # Показываем первые 3 примера
            if i < 3 and 'vae' in results and 'gan' in results:
                gen.show_comparison(
                    results['vae'],
                    results['gan'],
                    example['attrs'],
                    example['desc']
                )
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    # Создаем галерею всех результатов
    if all_results:
        create_gallery(all_results, gen, output_dir)
    
    return gen

def create_gallery(results, generator, output_dir):
    """Создание галереи всех результатов"""
    print("\n" + "=" * 70)
    print("📊 СОЗДАНИЕ ГАЛЕРЕИ РЕЗУЛЬТАТОВ")
    print("=" * 70)
    
    n = len(results)
    
    # Создаем большую фигуру
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    
    if n == 1:
        axes = axes.reshape(1, 2)
    
    for i, result in enumerate(results):
        # VAE
        vae_img = generator.tensor_to_image(result['vae'])
        axes[i, 0].imshow(vae_img)
        axes[i, 0].set_title(f"VAE: {result['desc']}", fontsize=11)
        axes[i, 0].axis('off')
        
        # GAN
        gan_img = generator.tensor_to_image(result['gan'])
        axes[i, 1].imshow(gan_img)
        axes[i, 1].set_title(f"GAN: {result['desc']}", fontsize=11)
        axes[i, 1].axis('off')
    
    plt.suptitle("Сравнение генераций VAE и GAN", fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Сохраняем галерею
    gallery_path = f"{output_dir}/gallery_comparison.png"
    plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
    print(f"\n📸 Галерея сохранена: {gallery_path}")
    
    # Показываем галерею
    print("\n👀 Открываю галерею...")
    img = Image.open(gallery_path)
    plt.figure(figsize=(14, 7))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Итоговая галерея: VAE vs GAN", fontsize=16)
    plt.show()

def demo_attribute_interpolation():
    """Демонстрация интерполяции атрибутов"""
    print("\n" + "=" * 70)
    print("🔄 ДЕМОНСТРАЦИЯ ИНТЕРПОЛЯЦИИ АТРИБУТОВ")
    print("=" * 70)
    
    gen = FaceGenerator()
    
    # Определяем начальные и конечные атрибуты
    start_attrs = [0, 0, 1, 1]  # Серьезный молодой мужчина без очков
    end_attrs = [1, 1, 0, 0]    # Улыбающаяся пожилая женщина в очках
    
    print(f"\nИнтерполяция от:")
    print(f"  Начало: {start_attrs} (Серьезный молодой мужчина)")
    print(f"  Конец:  {end_attrs} (Улыбающаяся пожилая женщина)")
    
    # Количество шагов
    num_steps = 7
    
    # Создаем папку для интерполяции
    interp_dir = "interpolation_demo"
    os.makedirs(interp_dir, exist_ok=True)
    
    print(f"\n🔄 Генерируем {num_steps} шагов интерполяции...")
    
    # Создаем фигуру для интерполяции
    fig, axes = plt.subplots(2, num_steps, figsize=(3 * num_steps, 6))
    
    for step in range(num_steps):
        # Линейная интерполяция атрибутов
        alpha = step / (num_steps - 1)
        interp_attrs = [
            (1 - alpha) * start + alpha * end 
            for start, end in zip(start_attrs, end_attrs)
        ]
        
        # Генерация
        results = gen.generate_face(
            attrs=interp_attrs,
            model='both',
            seed=step  # для разнообразия
        )
        
        # Сохранение
        if 'vae' in results:
            vae_img = gen.save_image(
                results['vae'],
                f"{interp_dir}/step_{step:02d}_vae.png"
            )
            axes[0, step].imshow(vae_img)
            axes[0, step].set_title(f"Step {step}\nVAE", fontsize=9)
            axes[0, step].axis('off')
        
        if 'gan' in results:
            gan_img = gen.save_image(
                results['gan'],
                f"{interp_dir}/step_{step:02d}_gan.png"
            )
            axes[1, step].imshow(gan_img)
            axes[1, step].set_title(f"Step {step}\nGAN", fontsize=9)
            axes[1, step].axis('off')
        
        print(f"  Шаг {step}: атрибуты {[round(a, 2) for a in interp_attrs]}")
    
    plt.suptitle("Интерполяция атрибутов: от мужчины к женщине", fontsize=14)
    plt.tight_layout()
    
    # Сохраняем интерполяцию
    interp_path = f"{interp_dir}/interpolation_complete.png"
    plt.savefig(interp_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Интерполяция сохранена: {interp_path}")
    plt.show()

def interactive_generation():
    """Интерактивная генерация"""
    print("\n" + "=" * 70)
    print("🎮 ИНТЕРАКТИВНАЯ ГЕНЕРАЦИЯ")
    print("=" * 70)
    
    gen = FaceGenerator()
    
    while True:
        print("\nВыберите действие:")
        print("1. Сгенерировать по шаблону")
        print("2. Ввести свои атрибуты")
        print("3. Выйти")
        
        choice = input("\nВаш выбор (1-3): ").strip()
        
        if choice == '1':
            print("\nШаблоны:")
            print("1. Улыбающийся молодой мужчина [1, 0, 1, 1]")
            print("2. Улыбающаяся молодая женщина [1, 0, 0, 1]")
            print("3. Серьезный мужчина в очках [0, 1, 1, 0]")
            print("4. Смешанные атрибуты [0.5, 0.5, 0.5, 0.5]")
            
            template = input("Выберите шаблон (1-4): ").strip()
            
            templates = {
                '1': [1, 0, 1, 1],
                '2': [1, 0, 0, 1],
                '3': [0, 1, 1, 0],
                '4': [0.5, 0.5, 0.5, 0.5]
            }
            
            if template in templates:
                attrs = templates[template]
                print(f"Генерирую с атрибутами: {attrs}")
                
                results = gen.generate_face(attrs, model='both')
                
                if 'vae' in results and 'gan' in results:
                    gen.show_comparison(
                        results['vae'],
                        results['gan'],
                        attrs,
                        "Сгенерированное лицо"
                    )
            else:
                print("Неверный выбор шаблона")
        
        elif choice == '2':
            print("\nВведите 4 числа (через пробел):")
            print("Улыбка (0-1), Очки (0-1), Пол (0=жен,1=муж), Возраст (0=стар,1=молод)")
            
            try:
                user_input = input("Атрибуты: ").strip()
                attrs = [float(x) for x in user_input.split()]
                
                if len(attrs) != 4:
                    print("Нужно ровно 4 числа!")
                    continue
                
                # Проверка диапазона
                for i, val in enumerate(attrs):
                    if val < 0 or val > 1:
                        print(f"Атрибут {i+1} должен быть от 0 до 1")
                        continue
                
                print(f"Генерирую с атрибутами: {attrs}")
                
                results = gen.generate_face(attrs, model='both')
                
                if 'vae' in results and 'gan' in results:
                    gen.show_comparison(
                        results['vae'],
                        results['gan'],
                        attrs,
                        "Ваше сгенерированное лицо"
                    )
                    
                    # Предлагаем сохранить
                    save = input("\nСохранить изображения? (y/n): ").strip().lower()
                    if save == 'y':
                        name = input("Имя файла (без расширения): ").strip()
                        if name:
                            os.makedirs("custom_generated", exist_ok=True)
                            gen.save_image(results['vae'], f"custom_generated/{name}_vae.png")
                            gen.save_image(results['gan'], f"custom_generated/{name}_gan.png")
                            print("✅ Изображения сохранены в custom_generated/")
            
            except ValueError:
                print("Ошибка: вводите только числа!")
        
        elif choice == '3':
            print("Выход...")
            break
        
        else:
            print("Неверный выбор")

def main():
    """Главная функция"""
    print("=" * 70)
    print("🤖 ГЕНЕРАТОР ЛИЦ НА ОСНОВЕ VAE И GAN")
    print("=" * 70)
    
    print("\nДоступные демонстрации:")
    print("1. Базовая генерация (рекомендуется для начала)")
    print("2. Интерполяция атрибутов")
    print("3. Интерактивная генерация")
    print("4. Все демонстрации")
    
    choice = input("\nВыберите вариант (1-4): ").strip()
    
    if choice == '1':
        demo_basic_generation()
    elif choice == '2':
        demo_attribute_interpolation()
    elif choice == '3':
        interactive_generation()
    elif choice == '4':
        demo_basic_generation()
        demo_attribute_interpolation()
        interactive_generation()
    else:
        print("Неверный выбор, запускаю базовую демонстрацию...")
        demo_basic_generation()
    
    print("\n" + "=" * 70)
    print("✨ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    print("\n📂 Сгенерированные изображения находятся в папках:")
    print("   - generated_faces/       (базовая генерация)")
    print("   - interpolation_demo/    (интерполяция)")
    print("   - custom_generated/      (интерактивная генерация)")

def quick_test():
    """Быстрый тест работы"""
    print("⚡ БЫСТРЫЙ ТЕСТ ГЕНЕРАЦИИ")
    
    gen = FaceGenerator()
    
    # Тестовые атрибуты
    test_attrs = [1, 0, 1, 1]  # Улыбающийся молодой мужчина
    
    print(f"\nГенерирую лицо с атрибутами: {test_attrs}")
    
    # Генерация
    results = gen.generate_face(test_attrs, model='both', seed=42)
    
    # Показ
    if 'vae' in results:
        gen.show_face(results['vae'], "VAE: Улыбающийся молодой мужчина")
    
    if 'gan' in results:
        gen.show_face(results['gan'], "GAN: Улыбающийся молодой мужчина")
    
    print("✅ Тест пройден успешно!")

if __name__ == "__main__":
    # Для быстрого теста (раскомментируйте):
    # quick_test()
    
    # Для полной демонстрации:
    main()